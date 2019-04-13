# encoding:utf-8

# Librairies tierces
import os

class Sequenceur:

    # Durees d'appui sur le bouton poussoir
    DUREE_APPUI_COURT_REDEMARRAGE = 2  # Nombre de secondes d'appui sur le poussoir pour reinitialiser le programme
    DUREE_APPUI_LONG_SHUTDOWN = 10  # Nombre de secondes d'appui sur le poussoir pour eteindre le raspberry

    tacho = 0
    cap_target = 0.0
    sequence = 0
    debut = True
    timeDebut = 0
    time = None
    programmeCourant = {}
    voiture = None
    asservissement = None

    timer_led = 0
    vitesse_clignote_led = 10
    led_clignote = True
    last_led = 0

    timer_bouton = 0
    last_bouton = 1  # 1 = bouton relache, 0 = bouton appuye
    flag_appui_court = False  # Passe a True quand un appui court (3 secondes) a ete detecte

    def __init__(self, voiture, time, arduino, asservissement, programme):
        self.programme = programme
        self.voiture = voiture
        self.time = time
        self.arduino = arduino
        self.asservissement = asservissement

    def execute(self):

        # Fait clignoter la led
        if self.led_clignote:
            if self.time.time() > self.timer_led + self.vitesse_clignote_led:
                self.timer_led = self.time.time()
                self.last_led = 0 if self.last_led else 1
                self.voiture.setLed(self.last_led)
        else:
            self.voiture.setLed(1)

        # Verifie appui court (3 sec) ou long (10 sec) sur bouton
        if self.voiture.getBoutonPoussoir() == 0:
            if self.last_bouton == 1:
                self.timer_bouton = self.time.time()
            else:
                if self.time.time() > self.timer_bouton + self.DUREE_APPUI_COURT_REDEMARRAGE:
                    # Arrete la voiture
                    self.voiture.avance(0)
                    self.voiture.tourne(0)
                    self.vitesse_clignote_led = 0.3
                    self.led_clignote = True
                    self.flag_appui_court = True
                if self.time.time() > self.timer_bouton + self.DUREE_APPUI_LONG_SHUTDOWN:
                    # Appui long: shutdown Raspberry Pi
                    os.system('sudo shutdown -h now')
                    pass
            self.last_bouton = 0
        else:
            self.last_bouton = 1
            if self.flag_appui_court:
                # Si on a detecte un appui court avant la relache du bouton
                self.flag_appui_court = False
                # Retourne a la sequence du debut
                for i in range(len(self.programme)):
                    if 'label' in self.programme[i]:
                        if self.programme[i]['label'] == 'attendBouton':
                            # On a trouve la prochaine sequence
                            self.sequence = i
                            self.debut = True

        if self.debut:
            # Premiere execution de l'instruction courante
            self.programmeCourant = self.programme[self.sequence]
            instruction = self.programmeCourant['instruction']
            print ("********** Nouvelle instruction *********** ", instruction)
            self.timeDebut = self.time.time()
            self.debut = False
            self.arduino.annuleRecalageCap()
            self.asservissement.cumulErreurCap = 0

            # Fait du cap courant le cap a suivre
            if instruction == 'setCap':
                target = self.arduino.getCap()
                print ('Cap Target : ', target)
                self.cap_target = target
                self.asservissement.setCapTarget(target)

            if instruction == 'setTacho':
                self.tacho = self.voiture.speedController.get_tacho()

            # Programme la vitesse de la voiture
            if instruction == 'ligneDroite' or \
                    instruction == 'tourne' or \
                    instruction == 'suiviLigne' or \
                    instruction == 'suiviImageLigneDroite' or \
                    instruction == 'suiviImageRoues' or \
                    instruction == 'suiviImageCap':
                vitesse = self.programmeCourant['vitesse']
                print ("Vitesse : ", vitesse)
                self.voiture.avance(vitesse)
                self.asservissement.setVitesse(vitesse)

            # Positionne les roues pour l'instruction 'tourne'
            if instruction == 'tourne':
                positionRoues = self.programmeCourant['positionRoues']
                print ("Position roues : ", positionRoues)
                self.voiture.tourne(positionRoues)

            # Ajoute une valeur a capTarget pour l'instruction 'ajouteCap'
            if instruction == 'ajouteCap':
                self.cap_target = (self.cap_target + self.programmeCourant['cap']) % 360
                self.asservissement.ajouteCap(self.programmeCourant['cap'])
                print ("Nouveau cap : ", self.cap_target)

            # Indique a la classe d'asservissement si elle doit asservir, et selon quel algo
            if instruction == 'ligneDroite':
                self.asservissement.initLigneDroite()
            elif instruction == 'suiviImageCap':
                self.asservissement.initSuiviImageCap()
            elif instruction == 'suiviImageRoues':
                self.asservissement.initSuiviImageRoues()
            elif instruction == 'suiviImageLigneDroite':
                activationDistanceIntegrale = False
                if 'activationDistanceIntegrale' in self.programmeCourant:
                    activationDistanceIntegrale = self.programmeCourant['activationDistanceIntegrale']
                self.asservissement.initSuiviImageLigneDroite(activationDistanceIntegrale)
            else:
                self.asservissement.annuleLigneDroite()
        else:
            # Partie qui s'execute en boucle tant que la condition de fin n'est pas remplie
            pass

        # Verifie s'il faut passer a l'instruction suivante
        finSequence = False  # Initialise finSequence
        # Recupere la condition de fin
        conditionFin = self.programmeCourant['conditionFin']
        # Verifie si la condition de fin est atteinte
        if conditionFin == 'attendreGyroStable':
            if self.arduino.gyroX != 0.0:
                # Si l'arduino a bien reussi a acquerir le gyro, le dit a travers la vitesse de clignotement de la led
                self.vitesse_clignote_led = 1.5
            finSequence = self.arduino.checkGyroStable()
        elif conditionFin == 'cap':
            capFinalMini = self.programmeCourant['capFinalMini']
            capFinalMaxi = self.programmeCourant['capFinalMaxi']
            if self.checkDeltaCapAtteint(capFinalMini, capFinalMaxi):
                finSequence = True
        elif conditionFin == 'duree':
            if (self.time.time() - self.timeDebut) > self.programmeCourant['duree']:
                finSequence = True
        elif conditionFin == 'tacho':
            print(self.voiture.speedController.get_tacho())
            if self.voiture.speedController.get_tacho() > (self.tacho + self.programmeCourant['tacho']):
                finSequence = True
        elif conditionFin == 'immediat':
            finSequence = True
        elif conditionFin == 'attendBouton':
            self.vitesse_clignote_led = 0.3
            self.led_clignote = True
            if self.voiture.getBoutonPoussoir() == 0:
                self.led_clignote = False
                finSequence = True

        if finSequence:
            # Si le champ nextLabel est defini, alors il faut chercher le prochain element par son label
            if 'nextLabel' in self.programmeCourant:
                nextLabel = self.programmeCourant['nextLabel']
                for i in range(len(self.programme)):
                    if 'label' in self.programme[i]:
                        if self.programme[i]['label'] == nextLabel:
                            # On a trouve la prochaine sequence
                            self.sequence = i
            else:
                # Si le champ nextLabel n'est pas defini, on passe simplement a l'element suivant
                self.sequence += 1
            self.debut = True

    def checkDeltaCapAtteint(self, capFinalMini, capFinalMaxi):
        absoluteCapMini = (self.cap_target + capFinalMini) % 360
        absoluteCapMaxi = (self.cap_target + capFinalMaxi) % 360

        ecartCapMini = (((self.arduino.getCap() - absoluteCapMini) + 180) % 360) - 180
        ecartCapMaxi = (((self.arduino.getCap() - absoluteCapMaxi) + 180) % 360) - 180

        if (ecartCapMini > 0 and ecartCapMaxi < 0):
            print ("--------------- Fin de virage ----------------")
            print ("CapTarget : ", self.cap_target, "Cap : ", self.arduino.getCap(), " Ecart cap mini : ", ecartCapMini,
                   " Ecart cap maxi : ", ecartCapMaxi)
            print ("----------------------------------------------")

        return (ecartCapMini > 0 and ecartCapMaxi < 0)

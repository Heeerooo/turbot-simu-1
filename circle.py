# Polynome

# y = ax² + bx + c


# Cercle

# (x - xmax)² + (y - ymax/2)² = r²


# Calcule l'angle depuis le robot vers l'intersection du polynome et du cercle (en radians)
import numpy as np


def angle_intersection(a, b, c, r, xmax=240, ymax=320):
    MIN_X_FOR_INTERSECTION = 10  # Si l'intersection a lieu plus bas que ce nombre de pixels, alors ne calcule pas l'intersection

    # Renvoie la coordonnée y pour un x donné selon le polynome

    def polynom(x):

        return a * x ** 2 + b * x + c

    # Calcule la distance au carré du robot vers un point du polynome défini par son ordonnée x

    def calcule_distance_carre(x):

        y = polynom(x)

        robot_x = xmax

        robot_y = ymax / 2

        distance_carre = (robot_x - x) ** 2 + (robot_y - y) ** 2

        return distance_carre

    # Return x on polynom that intersects circle

    def dichotomie(x=xmax // 2, step=xmax // 4):

        # Condition d'arrêt

        if step < 1:

            return x

        else:

            y = polynom(x)

            if ((x - xmax) ** 2 + (y - ymax / 2) ** 2) > r ** 2:

                return dichotomie(x + step, step // 2)

            else:

                return dichotomie(x - step, step // 2)

    # Find intersection point

    x = dichotomie()

    y = polynom(x)

    # Compute angle

    vx = xmax - x

    vy = y - ymax / 2

    # Ne prend pas en compte les lignes qui intersectent le cercle trop bas dans l'image

    if abs(vx) > MIN_X_FOR_INTERSECTION:

        angle = np.arctan(vy / vx)

    else:

        angle = None  # TODO renvoyer None et gérer le cas dans la fonction appelante ?

    return angle

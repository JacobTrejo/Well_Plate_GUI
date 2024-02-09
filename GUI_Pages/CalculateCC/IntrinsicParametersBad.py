import yaml
import warnings
import functools

# NOTE: it is important to import the librarys that you use to model the intrinsic parameters here,
#       some general ones have already been imported below.

import numpy as np
from numpy.random import normal as normrnd
from scipy.stats import norm
from scipy.stats import skewnorm

class IntrinsicParameters:

    """
        Class that obtains all the intrinsic parameter values from the IntrinsicParameters.yaml file
    """

    # belly_br_distribution = lambda: np.random.normal(0.405766627860074, 0.08510657640471547) * 400
    # belly_l_distribution = lambda: np.random.normal(1.0414934216489478, 0.0497660715368968)
    # belly_w_distribution = lambda: np.random.normal(0.4509085008872612, 0.04318248156070455)
    # c_belly_distribution = lambda: np.random.normal(0.8976990388969368, 0.046408648027029396)
    # c_eye_distribution = lambda: np.random.normal(1.7175761389389386, 0.20589859105033523)
    # c_head_distribution = lambda: skewnorm.rvs(13.160772264490056, 0.9469252531374381, 0.3251749094426817)
    # d_eye_distribution = lambda: np.random.normal(1.144396567090266, 0.04442945643226941)
    # eye_br_distribution = lambda: skewnorm.rvs(-8.231518385739356, 0.6969923960604942, 0.1853606969178898) * 400
    # eye_l_distribution = lambda: np.random.normal(0.5122961600634638, 0.07434852108369809)
    # eye_w_distribution = lambda: np.random.normal(0.4209238403150402, 0.040920763968052117)
    # head_br_distribution = lambda: np.random.normal(0.27670705295942777, 0.05571972183069393) * 400
    # head_l_distribution = lambda: np.random.normal(0.7889611717422198, 0.10205659701587598)
    # head_w_distribution = lambda: np.random.normal(0.5077558460625518, 0.032475097562161506)
    # seglen_distribution = lambda: 2 + np.random.rand()



    belly_br_distribution = lambda: np.random.normal(0.405766627860074, 0.08510657640471547) * 400
    # # Note the brightness here was kept as the original one
    belly_br_u = lambda: 235 * .83
    # belly_l_distribution = lambda: np.random.normal(1.0414934216489478, 0.0497660715368968)
    # belly_l_u = lambda: 1.0414934216489478
    # belly_w_distribution = lambda: np.random.normal(0.4509085008872612, 0.04318248156070455)
    # belly_w_u = lambda: 0.4509085008872612
    # c_belly_distribution = lambda: np.random.normal(0.8976990388969368, 0.046408648027029396)
    # c_belly_u = lambda: 0.8976990388969368
    # c_eye_distribution = lambda: np.random.normal(1.7175761389389386, 0.20589859105033523)
    # c_eye_u = lambda: 1.7175761389389386
    # c_head_distribution = lambda: skewnorm.rvs(13.160772264490056, 0.9469252531374381, 0.3251749094426817)
    # c_head_u = lambda: 0.9469252531374381
    # d_eye_distribution = lambda: np.random.normal(1.144396567090266, 0.04442945643226941)
    # d_eye_u = lambda: 1.144396567090266
    eye_br_distribution = lambda: skewnorm.rvs(-8.231518385739356, 0.6969923960604942, 0.1853606969178898) * 400
    # # Note the brightness was kept the same as the original
    eye_br_u = lambda: 235
    # eye_l_distribution = lambda: np.random.normal(0.5122961600634638, 0.07434852108369809)
    # eye_l_u = lambda: 0.5122961600634638
    # eye_w_distribution = lambda: np.random.normal(0.4209238403150402, 0.040920763968052117)
    # eye_w_u = lambda: 0.4209238403150402
    head_br_distribution = lambda: np.random.normal(0.27670705295942777, 0.05571972183069393) * 400
    head_br_u = lambda: .64 * .83 * 235
    # head_l_distribution = lambda: np.random.normal(0.7889611717422198, 0.10205659701587598)
    # head_l_u = lambda: 0.7889611717422198
    # head_w_distribution = lambda: np.random.normal(0.5077558460625518, 0.032475097562161506)
    # head_w_u = lambda: 0.5077558460625518
    # ball_thickness_distribution = lambda: np.random.normal(0.5861518680474747, 0.12086267623463233)
    # ball_thickness_u = lambda: 0.5861518680474747
    #
    # seglen_distribution = lambda: 2 + np.random.rand()
    # seglen_u = lambda: 2.5

    #####################################

    # Yaml file which controls the intrinsic parameters of the fish
    # NOTE: if you change a distribution make sure you also import the library in IntrinsicParameters.py

    # The new variables
    d_eye_distribution = lambda: np.random.normal(0.848341623974416, 0.09549722775699268)
    d_eye_u = lambda: 0.848341623974416
    c_eye_distribution = lambda: np.random.normal(1.7058500240221793, 0.10588899244079517)
    c_eye_u = lambda: 1.7058500240221793
    c_belly_distribution = lambda: np.random.normal(1.0815272808263487, 0.16799475269241487)
    c_belly_u = lambda: 1.0815272808263487
    c_head_distribution = lambda: np.random.normal(1.3516786182287277, 0.16174994860919834)
    c_head_u = lambda: 1.3516786182287277
    # eye_br_distribution = lambda: np.random.normal(0.3031255238836738 ,0.009569204555940871 )
    # eye_br_u = lambda: 0.3031255238836738
    # belly_br_distribution = lambda: np.random.normal(0.15215961514205165 ,0.007616258606638348 )
    # belly_br_u = lambda: 0.15215961514205165
    # head_br_distribution = lambda: np.random.normal(0.08987768639901683 ,0.014254666574661902 )
    # head_br_u = lambda: 0.08987768639901683

    eye_w_distribution = lambda: np.random.normal(0.3961904152425266, 0.015121026231446016)
    # eye_w_u = lambda: 0.5061904152425266
    eye_w_u = lambda: 0.3909238403150402

    # eye_w_distribution = lambda: np.random.normal(0.4209238403150402, 0.040920763968052117)
    # eye_w_u = lambda: 0.4209238403150402
    eye_l_distribution = lambda: np.random.normal(0.3844065772514719, 0.03507670474931163)
    # eye_l_u = lambda: 0.5144065772514719
    eye_l_u = lambda: 0.38

    belly_w_distribution = lambda: np.random.normal(0.4399572708604923, 0.005363904133890723)
    belly_w_u = lambda: 0.4399572708604923
    belly_l_distribution = lambda: np.random.normal(0.8692048085266663, 0.01026119884730811)
    belly_l_u = lambda: 0.8692048085266663
    head_w_distribution = lambda: np.random.normal(0.3155711134494867, 0.008015285933272387)
    head_w_u = lambda: 0.3155711134494867
    head_l_distribution = lambda: np.random.normal(0.713700736479039, 0.010046765180453425)
    head_l_u = lambda: 0.713700736479039
    # ball_size_distribution = lambda: np.random.normal(0.5127946718001636 ,0.1055655304428576 )
    # ball_size_u = lambda: 0.5127946718001636
    ball_thickness_distribution = lambda: np.random.normal(0.5861518680474747, 0.12086267623463233)
    ball_thickness_u = lambda: 0.5861518680474747
    # tail_brightness_distribution = lambda: np.random.normal(1.0064504977497606 ,0.010386653243516246 )
    # tail_brightness_u = lambda: 1.0064504977497606
    seglen_distribution = lambda: np.random.normal(3.906089250805046, 0.2831176584059629)
    seglen_u = lambda: 3.906089250805046

    def __init__(self, pathToYamlFile):

        """
            Essentially just a function to update the variables accordingly
        """

        static_vars = list(vars(IntrinsicParameters))[2:-3]

        file = open(pathToYamlFile, 'r')
        config = yaml.safe_load(file)
        keys = config.keys()
        list_of_vars_in_config = list(keys)

        # Updating the static variables
        for var in list_of_vars_in_config:
            if var in static_vars:
                value = config[var]
                line = 'IntrinsicParameters.' + var + ' = lambda : '

                line += str(value)
                try:
                    exec(line)
                except:
                    warnings.warn('\n' + line + ' could not be executed. \nYou might have forgotten to import the library in the Programs/IntrinsicParameters.py file \n' )
            else:
                warnings.warn(var + ' is not a valid variable, could be a spelling issue')


# IntrinsicParameters('inputs/IntrinsicParameters.yaml')
#IntrinsicParameters('inputs/IntrinsicParameters.yaml')

# print(IntrinsicParameters.c_head_distribution())

# l = []
# for _ in range(500):
#     l.append(IntrinsicParameters.c_head_distribution())
# l = np.array(l)
# print('mean: ', np.max(l))






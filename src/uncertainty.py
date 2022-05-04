import random


def determine_uncertainties(config):
    # Determine random start position for lander
    if not config['uncertainty'].get('start_position'):
        x = random.randrange(config['uncertainty']['start_positions_x_range'][0],
                             config['uncertainty']['start_positions_x_range'][1])
        y = random.randrange(config['uncertainty']['start_positions_y_range'][0],
                             config['uncertainty']['start_positions_y_range'][1])
        config['uncertainty']['start_position'] = x, y

    # Determine random start gravity of planet
    if not config['uncertainty'].get('gravity'):
        config['uncertainty']['gravity'] = random.randrange(config['uncertainty']['gravity_range'][1],
                                                            config['uncertainty']['gravity_range'][0])

    return config

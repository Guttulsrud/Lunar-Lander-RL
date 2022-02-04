def format_observation(observation):
    observation = observation['observation']
    return {

        'position': {
            'x': observation[0],
            'y': observation[1],
            'angle': observation[4],
        },
        'velocity': {
            'x': observation[2],
            'y': observation[3],
            'angular_velocity': observation[5],
        },

        'left_leg_contact': observation[6],
        'right_leg_contact': observation[7]
    }
import os

from loguru import logger

try:
    mode = os.environ['TRAINING_OR_INFERNCE']
except KeyError as e:
    raise ValueError('TRAINING_OR_INFERNCE environment variable is not set') from e

if mode == 'training':
    logger.info('Training mode!')
    from training import main

    main()

elif mode == 'inference':
    logger.info('Inference mode!')
    from inference import main

    main()

else:
    raise ValueError(
        f'Invalid mode: {mode}. It has to be either "training" or "inference"'
    )

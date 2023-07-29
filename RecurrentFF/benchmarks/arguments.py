import argparse
import logging


def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('--config-file',
                        type=str,
                        required=False,
                        help='Path to the config file')

    # Parse the arguments
    args = parser.parse_args()

    if args.config_file is not None:
        logging.info(f'Configured config file path is {args.config_file}')

    return args

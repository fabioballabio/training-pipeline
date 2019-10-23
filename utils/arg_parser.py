import argparse


class ArgParser:
    """Parses the cli arguments."""

    def __init__(self):
        super().__init__()
        self.parser = None
        self.arguments = None

    def parse_cli(self):
        """
        Parses the cli arguments.

            Returns:
                The path to the config_file.
        """

        # Initialize the parser
        # The description keyword, passed to the parser constructor is optional,
        # but allows us to add a brief description of the script when the help
        # message is displayed.
        self.parser = argparse.ArgumentParser(
            description="Specify the path to the config file"
        )

        # Now it"s time to add our first positional parameter to the script.
        # By default parames will be considered strings.
        # - and -- precede argument name
        # To speciy an optional argument, default contains the default value if nothing is specified
        # When argparse sees that a parameter is prefixed with hyphens assumes
        # it as optional.
        self.parser.add_argument(
            "-p",
            "--path",
            help="Path to json.config custom file",
            type=str,
            default="./configs/config.json",
            dest="path",
        )

        # Once we added our parameter, we must invoke the parse_args() method of the parser object.
        # This method will return an instance of the argparse.
        # Namespace class: the parsed parameters will be stored as attributes
        # of this instance.
        self.arguments = self.parser.parse_args()
        return self.arguments.path

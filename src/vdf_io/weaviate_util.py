from vdf_io.util import set_arg_from_input, set_arg_from_password


def prompt_for_creds(args):
    set_arg_from_input(
        args,
        "connection_type",
        "Enter 'local' or 'cloud' for connection types: ",
        choices=["local", "cloud"],
    )
    if args["connection_type"] == "cloud":
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance: ",
            str,
            env_var="WEAVIATE_URL",
        )
        set_arg_from_password(
            args,
            "api_key",
            "Enter the Weaviate API key: ",
            "WEAVIATE_API_KEY",
        )

    set_arg_from_password(
        args,
        "api_key",
        "Enter the Weaviate API key: ",
        "WEAVIATE_API_KEY",
    )

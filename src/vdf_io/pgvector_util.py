from vdf_io.util import set_arg_from_input, set_arg_from_password


def make_pgv_parser(DB_NAME_SLUG, subparsers):
    parser_pgvector = subparsers.add_parser(
        DB_NAME_SLUG, help="Import data to pgvector"
    )
    parser_pgvector.add_argument(
        "--platform",
        type=str,
        choices=["supabase", "neon", "tembo", "any"],
        default="any",
        help="Platform to connect to",
    )
    parser_pgvector.add_argument(
        "--schema",
        type=str,
        help="Schema of the Postgres instance (default: public)",
        default="public",
    )
    parser_pgvector.add_argument(
        "--connection_string",
        type=str,
        help="Connection string to Postgres instance",
    )
    parser_pgvector.add_argument("--user", type=str, help="User of Postgres instance")
    parser_pgvector.add_argument(
        "--password", type=str, help="Password of Postgres instance"
    )
    parser_pgvector.add_argument("--host", type=str, help="Host of Postgres instance")
    parser_pgvector.add_argument("--port", type=str, help="Port of Postgres instance")
    parser_pgvector.add_argument(
        "--dbname", type=str, help="Database name of Postgres instance"
    )
    parser_pgvector.add_argument(
        "--tables", type=str, help="Postgres tables to export (comma-separated)"
    )
    return parser_pgvector


def set_pgv_args_from_prompt(args):
    set_arg_from_input(
        args,
        "connection_string",
        "Enter the connection string to Postgres instance: ",
        str,
    )
    if not args.get("connection_string"):
        set_arg_from_input(
            args,
            "user",
            "Enter the user of Postgres instance (default: postgres): ",
            str,
            default_value="postgres",
        )
        set_arg_from_password(
            args,
            "password",
            "Enter the password of Postgres instance (default: postgres): ",
        )
        if not args.get("password"):
            # If password is not provided, set it to "postgres"
            args["password"] = "postgres"
        set_arg_from_input(
            args,
            "host",
            "Enter the host of Postgres instance: ",
            str,
            default_value="localhost",
        )
        set_arg_from_input(
            args,
            "port",
            "Enter the port of Postgres instance: ",
            str,
            default_value="5432",
        )
        set_arg_from_input(
            args,
            "dbname",
            "Enter the database name of Postgres instance: ",
            str,
            default_value="postgres",
        )

from awm.tools import tools_jsonl_load, normalize_scenario_name
from dataclasses import dataclass
import os
import sys
import textwrap
from loguru import logger

@dataclass
class Config:
    scenario: str
    envs_load_path: str # specify a path to load generated_envs.jsonl
    db_path: str | None = None # specify a path to load the database file, xxx.db
    host: str = "127.0.0.1"
    port: int = 8001
    temp_server_path: str | None = None # specify a temp server path, the code will be written to this path


    def pre_process(self):
        self.scenario = normalize_scenario_name(self.scenario)
        assert os.path.exists(self.envs_load_path), f"Environment file {self.envs_load_path} not found"
        if self.temp_server_path is None:
            folder = os.path.dirname(self.envs_load_path)
            self.temp_server_path = os.path.join(folder, f"temp_server_{self.scenario.lower()}.py")


def format_raw_code_to_lines(raw_code: str, indent: int) -> list[str]:
    no_indent_code = textwrap.dedent(raw_code).strip()
    indent_code = textwrap.indent(no_indent_code, ' ' * indent)
    return indent_code.split("\n")

def run_server(args: Config):
    envs = tools_jsonl_load(args.envs_load_path)
    envs = {normalize_scenario_name(e["scenario"]): e for e in envs}
    env = envs[args.scenario]
    
    if args.db_path is None:
        args.db_path = env["db_path"]
        logger.info(f"Using database file {args.db_path} from environment file")
    
    assert os.path.exists(args.db_path), f"Database file {args.db_path} not found"

    code = env["full_code"]
    new_code = [
        'import warnings',
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
        'from sqlalchemy.pool import NullPool',
    ]
    for line in code.split("\n"):

        if 'create_engine(' in line:
            left = line.split('create_engine(')[0]
            sql_path = f"'sqlite:///{args.db_path}'"
            # Use NullPool so that replacing the .db file between tasks works
            # without stale pooled connections pointing at the old file.
            right = f"create_engine({sql_path}, connect_args={{'check_same_thread': False}}, poolclass=NullPool)"
            line = f"{left}{right}"
                    
        if 'uvicorn.run(app' in line:
            raw_code = f"""
            import os
            host = os.environ.get('HOST', '{args.host}')
            port = os.environ.get('PORT', {args.port})
            print(f'Server starting on port={{port}}')                        
            """
            lines = format_raw_code_to_lines(raw_code, indent=4)
            raw_code = f"""
            from fastapi_mcp import FastApiMCP
            mcp = FastApiMCP(app)
            mcp.mount_http()
            print("MCP server enabled, please visit http://{args.host}:{args.port}/mcp for the MCP service")
            """
            lines += format_raw_code_to_lines(raw_code, indent=4)

            line = f'    uvicorn.run(app, host=host, port=int(port))'
            new_code.extend(lines)
            
        new_code.append(line)

    new_code = "\n".join(new_code)

    with open(args.temp_server_path, "w") as f:
        f.write(new_code)
    
    os.environ['PORT'] = str(args.port)
    os.environ['DATABASE_PATH'] = f"sqlite:///{args.db_path}"
    os.system(f"{sys.executable} {args.temp_server_path}")


def run(config: Config):
    run_server(config)


if __name__ == "__main__":
    from simpleArgParser import parse_args
    config: Config = parse_args(Config)
    run(config)

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from openenv.core.env_server.http_server import create_app

from models import SupportAction, SupportObservation


def _load_environment_class():
    try:
        from server.customer_support_environment import CustomerSupportEnvironment

        return CustomerSupportEnvironment
    except ModuleNotFoundError:
        # Fallback for runtimes where package-style imports are not resolved.
        env_file = Path(__file__).resolve().parent / "server" / "customer_support_environment.py"
        if not env_file.exists():
            raise

        spec = spec_from_file_location("customer_support_environment", env_file)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load customer_support_environment module")

        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CustomerSupportEnvironment


CustomerSupportEnvironment = _load_environment_class()

app = create_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="customer_support_benchmark",
    max_concurrent_envs=2,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
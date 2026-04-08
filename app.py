from __future__ import annotations

from openenv.core.env_server.http_server import create_app

from customer_support_environment import CustomerSupportEnvironment
from models import SupportAction, SupportObservation

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
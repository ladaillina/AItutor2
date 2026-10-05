import os
from fastapi_users import schemas

from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)

from dotenv import load_dotenv
from fastapi import Depends
from fastapi_users import BaseUserManager, IntegerIDMixin,FastAPIUsers
from fastapi_users.db import SQLAlchemyUserDatabase

from src.database import get_user_db
from src.models import User


load_dotenv()

SECRET = os.environ["AUTH_SECRET"]


class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
):
    yield UserManager(user_db)

bearer_transport = BearerTransport(
    tokenUrl="auth/jwt/login"
)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(
        secret=SECRET,
        lifetime_seconds=3600,
    )


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)


class UserRead(schemas.BaseUser[int]):
    pass


class UserCreate(schemas.BaseUserCreate):
    pass
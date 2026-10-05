from fastapi import FastAPI
from uuid import UUID

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from src.models import TutorMessage
from starlette.concurrency import run_in_threadpool

from src.database import get_db
from src.models import TutorSession
from src.sessions import create_session,load_session, delete_session
from pydantic import BaseModel, ConfigDict, Field
from fastapi import HTTPException
from pydantic import UUID4

from src.tutor import answer_question_jev

from src.auth import (
    fastapi_users,
    auth_backend,
    UserRead,
    UserCreate,
)

from src.auth import current_active_user
from src.models import User

app= FastAPI()

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)


class StudentMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=4000)


@app.get("/health") 
async def health():
   return {"status":"AITUTOr2 is running"}

@app.post("/sessions", status_code=201)
async def start_session(
    db: AsyncSession = Depends(get_db),user: User = Depends(current_active_user)
):
    session_id = create_session()

    db.add(TutorSession(id=UUID(session_id),user_id=user.id))

    await db.commit()

    return {"session_id": session_id}



@app.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: UUID4,
    payload: StudentMessage,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(current_active_user),
):
    session = await db.scalar(
        select(TutorSession).where(
            TutorSession.id == session_id,
            TutorSession.user_id == user.id,
        )
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        answer, _decision = await run_in_threadpool(
            answer_question_jev,
            payload.message,
            session_id=str(session_id),
        )
    except ValueError as exc:
        if str(exc) == "Session expired or unknown; create a new session.":
            raise HTTPException(
                status_code=404,
                detail="Session expired or not found",
            ) from exc
        raise

    db.add_all([
        TutorMessage(
            session_id=session_id,
            role="user",
            content=payload.message,
        ),
        TutorMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
        ),
    ])

    await db.commit()

    return {"reply": answer}


@app.get("/sessions/{session_id}")
async def get_session_history(
    session_id: UUID4,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(current_active_user),
):
    session = await db.scalar(
        select(TutorSession).where(
            TutorSession.id == session_id,
            TutorSession.user_id == user.id,
        )
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    result = await db.scalars(
        select(TutorMessage)
        .where(TutorMessage.session_id == session_id)
        .order_by(TutorMessage.id)
    )

    return {
        "session_id": str(session_id),
        "history": [
            {"role": msg.role, "content": msg.content}
            for msg in result.all()
        ],
    }


@app.delete("/sessions/{session_id}", status_code=204)
async def remove_session(
    session_id: UUID4,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(current_active_user),
):
    session = await db.scalar(
        select(TutorSession).where(
            TutorSession.id == session_id,
            TutorSession.user_id == user.id,
        )
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    await db.delete(session)
    await db.commit()

    await run_in_threadpool(delete_session, str(session_id))



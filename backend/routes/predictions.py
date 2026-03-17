"""Retention prediction endpoints (rule-based + ML)."""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.db_models import Topic, Review, QuizResult, User, Prediction
from models.schemas import PredictionOut
from services.auth_service import get_current_user
from services.decay_service import (
    rule_based_retention, ml_prediction, next_review_date,
    revision_recommendation, retention_label
)

from datetime import datetime, timezone

router = APIRouter()


@router.get("/{topic_id}", response_model=PredictionOut)
async def get_prediction(
    topic_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    topic = db.query(Topic).filter(Topic.id == topic_id, Topic.user_id == current_user.id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    review = db.query(Review).filter(Review.topic_id == topic_id, Review.user_id == current_user.id).first()
    latest_quiz = db.query(QuizResult).filter(
        QuizResult.topic_id == topic_id, QuizResult.user_id == current_user.id
    ).order_by(QuizResult.created_at.desc()).first()

    now = datetime.now(timezone.utc)
    last_reviewed = review.last_reviewed if review else topic.created_at
    if last_reviewed.tzinfo is None:
        last_reviewed = last_reviewed.replace(tzinfo=timezone.utc)

    review_count = review.review_count if review else 0
    quiz_score = latest_quiz.score if latest_quiz else 50
    hours_since = (now - last_reviewed).total_seconds() / 3600

    # Rule-based
    rule_ret = rule_based_retention(last_reviewed, quiz_score, review_count, topic.difficulty)

    # ML-based (async, may be None)
    ml_ret = await ml_prediction(hours_since, quiz_score, topic.difficulty, review_count, 30)

    # Next review date
    nr = next_review_date(review_count, last_reviewed)
    days_until = max(0, int((nr - now).total_seconds() / 86400))

    # Save prediction
    pred = Prediction(
        topic_id=topic_id,
        user_id=current_user.id,
        retention_rule_based=rule_ret,
        retention_ml=ml_ret,
    )
    db.add(pred)
    db.commit()

    return PredictionOut(
        topic_id=topic_id,
        topic_name=topic.name,
        retention_rule_based=rule_ret,
        retention_ml=ml_ret,
        recommendation=revision_recommendation(rule_ret, days_until),
        next_review_days=days_until,
        hours_since_review=round(hours_since, 1),
    )

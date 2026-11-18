import asyncio
from fastapi import APIRouter, HTTPException
from services.distress_service import llm_distress_score
from schemas.distress_score import DistressRequest
router = APIRouter()




@router.post("/api/distress-score")
async def distress_score_endpoint(request: DistressRequest):
    try:
        score = await asyncio.to_thread(
            llm_distress_score, request.text, request.conversationHistory
        )
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

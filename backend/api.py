from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/search")
def search(q: str = Query(..., description="Text query"), k: int = Query(5, le=50)):
    # TODO: wire to ml/search.py
    return JSONResponse([{"path": "example.jpg", "score": 0.0}])


@router.get("/change")
def change(before: str = Query(None), after: str = Query(None)):
    # TODO: wire to ml/change_detect.py
    return JSONResponse({"status": "stub", "change_percent": 0.0})


@router.get("/health")
def health():
    return {"status": "ok", "indexed": 0}

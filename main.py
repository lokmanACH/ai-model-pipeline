# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from grader import analyze_student_solution

# app = FastAPI(title="Gemini Exam Grader API")


# class GradingRequest(BaseModel):
#     exam_solution: str
#     student_solution: str


# class GradingResponse(BaseModel):
#     keyinsights: str
#     strengths: str
#     weaknesses: str
#     recommendation: str
#     rating: float
#     grade: float


# @app.get("/")
# def health():
#     return {"status": "ok", "description": "Gemini Exam Grader API"}


# @app.post("/grade", response_model=GradingResponse)
# def grade_exam(request: GradingRequest):
#     try:
#         result = analyze_student_solution(
#             exam_text=request.exam_solution,
#             student_text=request.student_solution
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     return result





from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from grader import analyze_student_solution  # your function
from typing import Optional

# ---------------- FastAPI ----------------
app = FastAPI(title="Gemini Exam Grader API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Accept any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Pydantic Schemas ----------------
class GradingRequest(BaseModel):
    exam_solution: str
    student_solution: str


class GradingResponse(BaseModel):
    keyinsights: Optional[str] = None
    strengths: Optional[str] = None
    weaknesses: Optional[str] = None
    recommendation: Optional[str] = None
    rating: Optional[float] = None
    grade: Optional[float] = None


# ---------------- Health Check ----------------
@app.get("/")
def health():
    return {"status": "ok", "description": "Gemini Exam Grader API"}


# ---------------- Grading Endpoint ----------------
@app.post("/grade", response_model=GradingResponse)
def grade_exam(request: GradingRequest):
    try:
        # Analyze the student solution using your Gemini function
        result = analyze_student_solution(
            exam_text=request.exam_solution,
            student_text=request.student_solution
        )

        # Ensure all keys exist, fill with None if missing
        keys = ["keyinsights", "strengths", "weaknesses", "recommendation", "rating", "grade"]
        for k in keys:
            if k not in result:
                result[k] = None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result

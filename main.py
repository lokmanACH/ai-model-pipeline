from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from grader import analyze_student_solution

app = FastAPI(title="Gemini Exam Grader API")


# ---------------- Request & Response Models ----------------
class GradingRequest(BaseModel):
    exam_solution: str
    student_solution: str
    api_key: str  # <-- include token here


class GradingResponse(BaseModel):
    keyinsights: str
    strengths: str
    weaknesses: str
    recommendation: str
    rating: float
    grade: float


# ---------------- Health Check ----------------
@app.get("/")
def health():
    return {"status": "ok", "description": "Gemini Exam Grader API"}


# ---------------- Grade Endpoint ----------------
@app.post("/grade", response_model=GradingResponse)
def grade_exam(request: GradingRequest):
    try:
        result = analyze_student_solution(
            exam_text=request.exam_solution,
            student_text=request.student_solution,
            api_key=request.api_key  # <-- pass token from request
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result



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

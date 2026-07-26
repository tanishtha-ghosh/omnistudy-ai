from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import uuid

from backend.rag_engine import rag_engine
from backend.sample_data import load_sample_os_pdf

app = FastAPI(title="OmniStudy AI Backend", version="1.2.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared In-Memory Data Store (Extensible to PostgreSQL / SQLite)
db = {
    "profile": {
        "id": "usr_101",
        "name": "Alex Rivers",
        "avatar_url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&auto=format&fit=crop&q=80",
        "email": "alex.rivers@university.edu",
        "college": "Stanford University",
        "degree": "B.Tech Computer Science",
        "semester": "Semester 5",
        "subjects": ["Operating Systems", "Database Systems", "Computer Networks", "Artificial Intelligence"],
        "goals": "Maintain a 3.9 GPA and master OS page replacement & RAG systems.",
        "study_hours": "3.5"
    },
    "documents": [],
    "assignments": [
        {"id": "asg_1", "title": "Operating Systems Lab 4 (Page Fault Simulator)", "course": "CS 301 - OS", "due_date": "2026-07-29", "status": "In Progress", "priority": "High"},
        {"id": "asg_2", "title": "Database Systems Quiz 2 Revision", "course": "CS 302 - DBMS", "due_date": "2026-07-31", "status": "To Do", "priority": "Medium"},
        {"id": "asg_3", "title": "Computer Networks Essay on TCP/IP", "course": "CS 304 - CN", "due_date": "2026-08-03", "status": "Completed", "priority": "Low"}
    ],
    "exams": [
        {"id": "ex_1", "title": "Operating Systems Midterm", "course": "CS 301", "date": "2026-08-05", "days_left": 10},
        {"id": "ex_2", "title": "Database Systems Final", "course": "CS 302", "date": "2026-08-12", "days_left": 17}
    ],
    "study_plan": [
        {"id": "slot_1", "time": "09:00 AM - 10:30 AM", "topic": "Page Replacement Algorithms (FIFO vs LRU)", "course": "CS 301 - OS", "status": "Completed"},
        {"id": "slot_2", "time": "11:00 AM - 12:30 PM", "topic": "Belady's Anomaly & Working Set Model", "course": "CS 301 - OS", "status": "Pending"},
        {"id": "slot_3", "time": "02:00 PM - 03:30 PM", "topic": "Practice 10 Quiz Questions on Page Faults", "course": "CS 301 - OS", "status": "Pending"},
        {"id": "slot_4", "time": "04:30 PM - 05:30 PM", "topic": "Review DBMS Indexing Flashcards", "course": "CS 302 - DBMS", "status": "Pending"}
    ]
}


@app.on_event("startup")
def startup_event():
    """Load pre-populated sample Operating Systems PDF into RAG engine on start."""
    try:
        if not db["documents"]:
            metadata, pdf_bytes = load_sample_os_pdf()
            db["documents"].append({
                "id": metadata["doc_id"],
                "filename": metadata["filename"],
                "page_count": metadata["page_count"],
                "total_words": metadata["total_words"],
                "chunk_count": metadata["chunk_count"],
                "uploaded_at": "Just now (Pre-loaded Sample)"
            })
    except Exception as e:
        print(f"Error loading sample PDF: {e}")


# --- Pydantic Models ---
class ChatQuery(BaseModel):
    doc_id: str
    query: str
    api_key: Optional[str] = None

class AssignmentItem(BaseModel):
    title: str
    course: str
    due_date: str
    status: str = "To Do"
    priority: str = "Medium"

class ProfileUpdateModel(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(...)
    college: str = Field(...)
    degree: str = Field(...)
    semester: str = Field("Semester 5")
    avatar_url: Optional[str] = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&auto=format&fit=crop&q=80"
    subjects: Optional[List[str]] = []
    goals: Optional[str] = ""
    study_hours: Optional[str] = "3.5"

class StudySlotModel(BaseModel):
    id: Optional[str] = None
    time: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    course: str = Field(..., min_length=1)
    status: str = Field("Pending")


# --- API Routes ---

@app.get("/api/profile")
def get_profile():
    return db["profile"]

@app.post("/api/profile")
@app.put("/api/profile")
def update_profile(profile_data: ProfileUpdateModel):
    updated = {
        "id": db["profile"].get("id", "usr_101"),
        "name": profile_data.name.strip(),
        "avatar_url": profile_data.avatar_url or "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&auto=format&fit=crop&q=80",
        "email": profile_data.email.strip(),
        "college": profile_data.college.strip(),
        "degree": profile_data.degree.strip(),
        "semester": profile_data.semester.strip(),
        "subjects": profile_data.subjects if profile_data.subjects else ["Operating Systems", "Database Systems"],
        "goals": profile_data.goals.strip(),
        "study_hours": str(profile_data.study_hours or "3.5")
    }
    db["profile"] = updated
    return {"message": "Profile updated successfully", "profile": updated}


@app.get("/api/dashboard")
def get_dashboard():
    profile = db["profile"]
    return {
        "user": profile,
        "documents": db["documents"],
        "assignments": db["assignments"],
        "exams": db["exams"],
        "today_study_plan": db["study_plan"],
        "stats": {
            "notes_uploaded": len(db["documents"]),
            "pending_assignments": sum(1 for a in db["assignments"] if a["status"] != "Completed"),
            "upcoming_exams": len(db["exams"]),
            "study_hours_today": profile.get("study_hours", "3.5")
        }
    }


# --- Study Planner Editing & Status Toggle Endpoints ---

@app.get("/api/planner/slots")
def get_planner_slots():
    return db["study_plan"]

@app.post("/api/planner/slots")
def add_study_slot(slot: StudySlotModel):
    """Add a new custom study slot to the plan."""
    new_slot = {
        "id": f"slot_{uuid.uuid4().hex[:6]}",
        "time": slot.time,
        "topic": slot.topic,
        "course": slot.course,
        "status": slot.status
    }
    db["study_plan"].append(new_slot)
    return {"message": "Study slot added successfully", "slot": new_slot, "study_plan": db["study_plan"]}

@app.put("/api/planner/slots/{slot_id}")
def update_study_slot(slot_id: str, slot: StudySlotModel):
    """Edit topic, time, course, or status of an existing slot."""
    for item in db["study_plan"]:
        if item["id"] == slot_id:
            item["time"] = slot.time
            item["topic"] = slot.topic
            item["course"] = slot.course
            item["status"] = slot.status
            return {"message": "Slot updated successfully", "slot": item, "study_plan": db["study_plan"]}
    raise HTTPException(status_code=404, detail="Study slot not found")

@app.put("/api/planner/slots/{slot_id}/toggle")
def toggle_slot_status(slot_id: str):
    """Toggle status between Completed and Pending."""
    for item in db["study_plan"]:
        if item["id"] == slot_id:
            item["status"] = "Completed" if item["status"] != "Completed" else "Pending"
            return {"message": f"Slot status changed to {item['status']}", "slot": item, "study_plan": db["study_plan"]}
    raise HTTPException(status_code=404, detail="Study slot not found")

@app.delete("/api/planner/slots/{slot_id}")
def delete_study_slot(slot_id: str):
    """Delete a study slot from the plan."""
    db["study_plan"] = [s for s in db["study_plan"] if s["id"] != slot_id]
    return {"message": "Slot deleted successfully", "study_plan": db["study_plan"]}

@app.post("/api/planner/generate")
def generate_planner(hours_per_day: float = 3.0):
    """AI Timetable Generator."""
    user_subjects = db["profile"].get("subjects", ["Operating Systems", "Database Systems"])
    sub1 = user_subjects[0] if len(user_subjects) > 0 else "Operating Systems"
    sub2 = user_subjects[1] if len(user_subjects) > 1 else "Database Systems"

    schedule = [
        {"id": "slot_gen_1", "time": "09:00 AM - 10:30 AM", "topic": f"{sub1}: Core Concepts & Page Replacement Algorithms", "course": db["profile"].get("degree", "CS"), "status": "Pending"},
        {"id": "slot_gen_2", "time": "11:00 AM - 12:15 PM", "topic": f"{sub1}: Advanced Exercises & Working Set Model", "course": db["profile"].get("degree", "CS"), "status": "Pending"},
        {"id": "slot_gen_3", "time": "02:00 PM - 03:30 PM", "topic": f"{sub2}: B+ Tree Indexing & Query Optimization", "course": db["profile"].get("degree", "CS"), "status": "Pending"},
        {"id": "slot_gen_4", "time": "04:00 PM - 05:00 PM", "topic": "Exam Preparation & Practice 10 Quiz MCQs", "course": db["profile"].get("degree", "CS"), "status": "Pending"}
    ]
    db["study_plan"] = schedule
    return {"message": "New study timetable generated successfully", "study_plan": schedule}


# --- PDF & Chat Endpoints ---

@app.post("/api/documents/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    contents = await file.read()
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    metadata = rag_engine.process_pdf(doc_id, file.filename, contents)
    
    doc_entry = {
        "id": metadata["doc_id"],
        "filename": metadata["filename"],
        "page_count": metadata["page_count"],
        "total_words": metadata["total_words"],
        "chunk_count": metadata["chunk_count"],
        "uploaded_at": "Just now"
    }
    db["documents"].insert(0, doc_entry)
    return {"message": "PDF uploaded and processed successfully", "document": doc_entry}

@app.get("/api/documents")
def get_documents():
    return db["documents"]

@app.post("/api/chat")
def chat_with_notes(data: ChatQuery):
    if not data.doc_id:
        raise HTTPException(status_code=400, detail="Please select a document to chat with.")
    return rag_engine.generate_answer(data.query, data.doc_id, api_key=data.api_key)

@app.post("/api/documents/{doc_id}/summarize")
def summarize_doc(doc_id: str, api_key: Optional[str] = None):
    return rag_engine.generate_summary(doc_id, api_key=api_key)

@app.post("/api/documents/{doc_id}/quiz")
def generate_quiz(doc_id: str, count: int = 5, api_key: Optional[str] = None):
    quiz_questions = rag_engine.generate_quiz(doc_id, count=count, api_key=api_key)
    return {"doc_id": doc_id, "questions": quiz_questions}

@app.post("/api/documents/{doc_id}/flashcards")
def generate_flashcards(doc_id: str, api_key: Optional[str] = None):
    cards = rag_engine.generate_flashcards(doc_id, api_key=api_key)
    return {"doc_id": doc_id, "flashcards": cards}

@app.get("/api/assignments")
def get_assignments():
    return db["assignments"]

@app.post("/api/assignments")
def add_assignment(item: AssignmentItem):
    new_asg = {
        "id": f"asg_{uuid.uuid4().hex[:6]}",
        "title": item.title,
        "course": item.course,
        "due_date": item.due_date,
        "status": item.status,
        "priority": item.priority
    }
    db["assignments"].append(new_asg)
    return new_asg

@app.put("/api/assignments/{asg_id}/status")
def update_assignment_status(asg_id: str, status: str):
    for asg in db["assignments"]:
        if asg["id"] == asg_id:
            asg["status"] = status
            return asg
    raise HTTPException(status_code=404, detail="Assignment not found")


# --- Static Frontend Serving ---
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>OmniStudy AI API Running</h1>"

import fitz  # PyMuPDF
from backend.rag_engine import rag_engine

def load_sample_os_pdf():
    """Create a sample Operating Systems lecture note PDF and process it in RAG Engine."""
    doc = fitz.open()  # New PDF
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Operating Systems - Chapter 3: Page Replacement Algorithms", fontsize=16)
    page1.insert_text((50, 90), 
"""Virtual memory allows an operating system to execute processes that exceed physical RAM size.
When a page fault occurs and no free memory frame is available, the operating system must choose
a page in RAM to evict to disk and replace with the requested page from secondary storage.

Page Replacement Algorithms evaluate which page frame to swap out to minimize future page faults.

1. First-In, First-Out (FIFO)
FIFO tracks all pages in memory in a queue. The page at the head of the queue (the oldest page)
is evicted first. While simple to implement using a pointer or queue data structure, FIFO suffers
from Belady's Anomaly—where increasing the number of allocated memory frames unexpectedly leads
to an increased number of page faults for certain reference strings.""", fontsize=11)

    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Chapter 3 (Continued): LRU and Optimal Page Replacement", fontsize=16)
    page2.insert_text((50, 90), 
"""2. Optimal Page Replacement (OPT / MIN)
The Optimal algorithm replaces the page that will not be used for the longest period of time in the future.
For example, if pages A, B, C are in memory and reference sequence is A, B, D, C, replacing C is optimal.
Although OPT guarantees the lowest possible page fault rate for any frame allocation, it requires
future knowledge of process execution, making it impossible to implement in real-time operating systems.
It serves primarily as a theoretical benchmark.

3. Least Recently Used (LRU)
LRU replaces the page that has not been accessed for the longest duration in past execution.
LRU uses past behavior as an approximation of future access patterns.
Implementation approaches:
- Counter (Clock/Timestamp approach): Every page table entry has a time-of-use field updated on every reference.
- Stack approach: A stack of page numbers is maintained; when a page is referenced, it moves to the top.""", fontsize=11)

    # Page 3
    page3 = doc.new_page()
    page3.insert_text((50, 50), "Chapter 3: Second Chance, Thrashing & Memory Management", fontsize=16)
    page3.insert_text((50, 90), 
"""4. Second Chance (Clock) Algorithm
The Clock algorithm approximates LRU with lower hardware overhead. It arranges frames in a circular queue
with a single reference bit per frame. When a page is referenced, hardware sets its reference bit to 1.
During page replacement, the clock hand inspects frames:
- If reference bit is 1, it is cleared to 0 and given a 'second chance'.
- If reference bit is 0, that page is selected for immediate replacement.

Thrashing & Working Set Model
Thrashing occurs when a process spends more time swapping pages in and out of swap space than executing instructions.
This occurs when the total demand for pages across active processes exceeds available physical frames.
The Working Set Model defines the set of pages actively referenced during a moving time window W(t, delta).
To prevent thrashing, the OS suspended processes if total working set size exceeds total RAM frames.""", fontsize=11)

    pdf_bytes = doc.tobytes()
    doc_id = "doc_sample_os_notes"
    filename = "Operating Systems - Chapter 3 (Page Replacement Algorithms).pdf"
    
    metadata = rag_engine.process_pdf(doc_id, filename, pdf_bytes)
    return metadata, pdf_bytes

import json
from pathlib import Path

from src.retrieval import retrieve


QUERIES = [
    # Real Numbers
    "What is the Fundamental Theorem of Arithmetic?",
    "Explain Euclid's division lemma.",
    "Why is sqrt 3 irrational?",
    "What is the relation between HCF and LCM of two positive integers?",
    "What does Theorem 1.2 say if a prime p divides a squared?",

    # Polynomials
    "What is the relationship between zeroes and coefficients of a quadratic polynomial?",
    "Find the zeroes of x^2 - 7x + 10.",
    "What is a polynomial with two zeroes?",
    
    # Pair of Linear Equations
    "When does a pair of linear equations have no solution?",
    "What does a pair of intersecting lines mean for two linear equations?",
    "How do I solve two linear equations by elimination?",
    "When are two linear equations inconsistent?",

    # Quadratic Equations
    "What is the discriminant of a quadratic equation?",
    "How do I know if a quadratic equation has equal roots?",
    "Explain the quadratic formula.",
    "What happens when b squared minus 4ac is negative?",

    # Arithmetic Progressions
    "What is the nth term of an arithmetic progression?",
    "How do I find the sum of the first n terms of an AP?",
    "How can I find a missing term in an arithmetic progression?",
    "What is the common difference of an AP?",

    # Triangles
    "State the Basic Proportionality Theorem.",
    "What are the conditions for two triangles to be similar?",
    "Explain the converse of the Pythagoras theorem.",
    "Why are corresponding sides proportional in similar triangles?",

    # Coordinate Geometry
    "What is the distance formula between two points?",
    "How do I find the point dividing a line segment in a given ratio?",
    "Find the distance between two points on the coordinate plane.",

    # Trigonometry
    "What are the trigonometric ratios of an acute angle?",
    "What is the identity sin squared theta plus cos squared theta?",
    "What are the values of sin 30 and cos 60?",
    "If tan theta is known how can I find sin theta and cos theta?",

    # Applications of Trigonometry
    "How do I solve a height and distance problem?",
    "What is the angle of elevation?",
    "A tower casts a shadow. How can I find its height using trigonometry?",

    # Circles
    "Why is a tangent perpendicular to the radius at the point of contact?",
    "What can we say about two tangents drawn from the same external point?",
    "Explain the tangent theorem for a circle.",

    # Areas Related to Circles
    "How do I find the area of a sector of a circle?",
    "What is the difference between a sector and a segment of a circle?",
    "How do I calculate the length of an arc?",

    # Surface Areas and Volumes
    "How do I find the volume of a cone?",
    "How do I calculate the surface area of a combination of solids?",
    "What is the volume of a sphere?",

    # Statistics
    "How do I calculate the mean of grouped data?",
    "How do I find the median of grouped data?",
    "What is the mode of grouped data?",
    "What is a class mark in statistics?",

    # Probability
    "What is the probability of an impossible event?",
    "What is the probability of a certain event?",
    "How do I find the probability of getting a particular outcome?",
]


OUTPUT = Path("artifacts/retrieval_50.jsonl")
OUTPUT.parent.mkdir(exist_ok=True)


with OUTPUT.open("w", encoding="utf-8") as f:
    for query_number, query in enumerate(QUERIES, start=1):
        print(f"\n[{query_number}/50] {query}")

        results = retrieve(query)

        record = {
            "query_number": query_number,
            "query": query,
            "results": [],
        }

        for rank, point in enumerate(results, start=1):
            payload = point.payload

            result = {
                "rank": rank,
                "score": point.score,
                "chapter": payload.get("chapter"),
                "title": payload.get("title"),
                "content": payload.get("content"),
                "document_id": payload.get("document_id"),
                "parent_id": payload.get("parent_id"),
            }

            record["results"].append(result)

            # Keep terminal output compact.
            if rank <= 3:
                print(
                    f"  {rank}. "
                    f"{payload.get('chapter')} | "
                    f"{payload.get('title')} | "
                    f"{point.score:.4f}"
                )

        f.write(json.dumps(record, ensure_ascii=False) + "\n")


print(f"\nSaved results to: {OUTPUT}")
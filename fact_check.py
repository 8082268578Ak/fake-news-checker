import requests

GOOGLE_API_KEY = "YOUR_API_KEY"

def google_fact_check(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={GOOGLE_API_KEY}"

    response = requests.get(url).json()

    claims = response.get("claims", [])

    if not claims:
        return "no_evidence", []

    results = []

    for claim in claims[:3]:
        review = claim.get("claimReview", [{}])[0]

        publisher = review.get("publisher", {}).get("name", "Unknown")
        rating = review.get("textualRating", "No rating")
        link = review.get("url", "#")

        results.append({
            "publisher": publisher,
            "rating": rating,
            "url": link
        })

    return "verified", results
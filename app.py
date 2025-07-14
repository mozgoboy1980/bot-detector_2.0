from flask import Flask, request, jsonify, render_template
import re, time
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from googleapiclient.discovery import build

YOUTUBE_API_KEY = "AIzaSyDqxbv4cFaujBl6oY42_KJzWXA1SD93cWw"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
app = Flask(__name__, template_folder="templates")

def preprocess_text(comment):
    text = re.sub(r"http\S+|www\S+|t.me/\S+", "", comment.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

BOT_PHRASES = ["дзякуй за працу", "малады чалавек", "жыве беларусь", "так і трэба",
               "усе правільна сказаў", "дастане усіх", "лукашэнка малайчына",
               "бот", "тролі", "выгнанцы", "пятай калоне", "купленыя заходам"]

def keyword_score(text): return sum(1 for p in BOT_PHRASES if p in text)

def behavioral_flags(meta):
    f = 0
    if meta.get("channel_age_days", 9999) < 7: f += 1
    if meta.get("subscriber_count", 1) == 0: f += 1
    if meta.get("comment_count", 0) > 100 and meta.get("video_count", 1) == 0: f += 1
    return f

def cluster_comments(texts):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    emb = model.encode(texts)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(emb)
    return dict(zip(texts, clustering.labels_))

def final_score(comment, meta, clusters=None):
    text = preprocess_text(comment)
    score = keyword_score(text) + behavioral_flags(meta)
    if clusters and clusters.get(text, -1) != -1: score += 1
    return score

def is_bot(score): return score >= 3

def get_comments(video_id, max_results=100):
    out, token = [], None
    while len(out) < max_results:
        r = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, pageToken=token).execute()
        for i in r["items"]:
            sn = i["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "text": sn["textDisplay"],
                "author": sn["authorChannelId"].get("value", ""),
                "published": sn["publishedAt"]
            })
        token = r.get("nextPageToken")
        if not token: break
        time.sleep(0.5)
    return out

def get_channel_metadata(cid):
    try:
        r = youtube.channels().list(part="snippet,statistics", id=cid).execute()
        if not r["items"]: return {}
        d = r["items"][0]
        s, sn = d["statistics"], d["snippet"]
        created = time.mktime(time.strptime(sn["publishedAt"], "%Y-%m-%dT%H:%M:%S.%fZ"))
        return {
            "subscriber_count": int(s.get("subscriberCount", 0)),
            "video_count": int(s.get("videoCount", 0)),
            "comment_count": int(s.get("commentCount", 0)),
            "channel_age_days": (time.time() - created) / 86400
        }
    except: return {}

def analyze_video(video_id):
    comments = get_comments(video_id)
    texts = [preprocess_text(c["text"]) for c in comments]
    clusters = cluster_comments(texts)
    results = []
    for c in comments:
        meta = get_channel_metadata(c["author"])
        score = final_score(c["text"], meta, clusters)
        results.append({
            "text": c["text"], "author": c["author"], "score": score,
            "probability": min(round(score / 5, 2), 1.0), "is_bot": is_bot(score),
            "metadata": meta
        })
    return results

@app.route("/")
def index(): return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    video_id = request.json.get("video_id")
    return jsonify(analyze_video(video_id))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

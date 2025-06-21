import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
import os

# Load the CSV
df = pd.read_csv("data.csv")
feedback = pd.read_csv("feedback.csv")

# Convert Date and Time columns to datetime
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date

# Create a directory to save plots
os.makedirs("plots", exist_ok=True)

# Set up PDF
pdf = SimpleDocTemplate("Chatbot_Report.pdf", pagesize=A4)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='EmojiText', fontName='Helvetica', fontSize=12, leading=16, alignment=TA_LEFT))
story = []

def save_plot(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    story.append(Image(filename, width=6*inch, height=3.5*inch))
    story.append(Spacer(1, 8))

# 1. Top 10 FAQs by frequency
top_faqs = df['User Question'].value_counts().head(10)
truncated_labels = [q[:50] + '...' if len(q) > 60 else q for q in top_faqs.index]

story.append(Paragraph(f"<b>Top {top_faqs.shape[0]} FAQs:</b>", styles['EmojiText']))
for q, count in top_faqs.items():
    story.append(Paragraph(f"Q: {q}<br/>Asked: {count} times<br/><br/>", styles['EmojiText']))
story.append(Spacer(1, 12))

# Bar plot for plotting the top FAQ's
fig1 = plt.figure(figsize=(12, 6))
sns.barplot(x=top_faqs.values, y=truncated_labels)
plt.title("Top 10 Most Asked Questions")
plt.xlabel("Frequency")
plt.ylabel("User Question")
plt.tight_layout()
save_plot(fig1, "plots/top_faqs.png")

# 2. Time taken stats
time_taken = df['Time Taken(in sec)']
min_time = time_taken.min()
max_time = time_taken.max()
min_row = df[df['Time Taken(in sec)'] == min_time].iloc[0]
max_row = df[df['Time Taken(in sec)'] == max_time].iloc[0]

story.append(Paragraph("<b>Time Taken Stats:</b>", styles['EmojiText']))
story.append(Paragraph(f"Average: {time_taken.mean():.2f} seconds<br/>"
                       f"Minimum: {min_time:.2f} sec → {min_row['User Question']}<br/>"
                       f"Maximum: {max_time:.2f} sec → {max_row['User Question']}", styles['EmojiText']))
story.append(Spacer(1, 12))

# Histogram showing the time taken to answer queries
fig2 = plt.figure(figsize=(8, 5))
sns.histplot(df['Time Taken(in sec)'], bins=20, color='mediumvioletred')
plt.title("Distribution of Response Time")
plt.xlabel("Time Taken (in seconds)")
plt.ylabel("Frequency")
plt.tight_layout()
save_plot(fig2, "plots/response_time.png")

# 3. Count of unanswered queries
unanswered_text = "Sorry, we can't get the answer of the question. \n\n-Please again ask the same question more precisely. \n\n-Contact us on Email - librarian@iitgn.ac.in or Phone Number - +91-079-2395-2431"
busy_text = "Thanks for connecting! All the agents are currently busy. Please try again after 1 minute.\n\nWhile you wait, feel free to browse our Library Website - https://library.iitgn.ac.in/ or Search the Catalogue - https://catalog.iitgn.ac.in/."

unanswered_count = 0
busy_count = 0
correct_count = 0
unanswered_queries = []
busy_queries = []
for i in range(len(df)):
    if (df['Chatbot Answer'][i] == unanswered_text):
        unanswered_count += 1
        unanswered_queries.append(df['User Question'][i])
    elif (df['Chatbot Answer'][i] == busy_text):
        busy_count += 1
        busy_queries.append(df['User Question'][i])
    else:
        correct_count += 1

story.append(Paragraph(f"<b>Unanswered Queries Count:</b> {unanswered_count}", styles['EmojiText']))
for q in unanswered_queries:
    story.append(Paragraph(f"→ {q}", styles['EmojiText']))
story.append(Spacer(1, 12))

story.append(Paragraph(f"<b>Busy Queries Count:</b> {busy_count}", styles['EmojiText']))
for q in busy_queries:
    story.append(Paragraph(f"→ {q}", styles['EmojiText']))
story.append(Spacer(1, 12))

labels = []
values = []

if correct_count > 0:
    labels.append('Valid Answers')
    values.append(correct_count)

if unanswered_count > 0:
    labels.append('Unanswered')
    values.append(unanswered_count)

if busy_count > 0:
    labels.append('Busy chatbot')
    values.append(busy_count)

# Pie chart to show the percentage of queries answered/ unanswered / bot busy
fig3 = plt.figure(figsize=(6, 4))
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Distribution of Responses")
plt.tight_layout()
save_plot(fig3, "plots/response_pie.png")

# 4. Busiest time of day (by hour)
busiest_hour = df['Hour'].value_counts().idxmax()

story.append(Paragraph(f"<b>Most Busy Hour:</b> {busiest_hour}:00 - {busiest_hour + 1}:00", styles['EmojiText']))
story.append(Spacer(1, 12))

df['Hour'] = pd.to_datetime(df['Time'], format="%H:%M:%S").dt.hour
hourly_counts = df['Hour'].value_counts().sort_index()

# Scatter plot showing at what time of the day the queries were asked
fig4 = plt.figure(figsize=(10, 6))
sns.scatterplot(x=hourly_counts.index, y=hourly_counts.values, marker='o')
plt.title("Number of Queries by Hour of Day")
plt.xlabel("Hour (24-hour format)")
plt.ylabel("Number of Queries")
plt.grid(True)
plt.xticks(range(0, 24))
plt.tight_layout()
save_plot(fig4, "plots/hourly.png")

# 5. Daily Queries count for last 30 days
last_30_days = datetime.now().date() - timedelta(days=30)
recent_df = df[df['Date'] >= pd.to_datetime(last_30_days).date()]
daily_counts = recent_df['Date'].value_counts().sort_index()

# Bar plot showing the number of queries asked each day
fig5 = plt.figure(figsize=(12, 6))
daily_counts.plot(kind='bar', color='green')
plt.title("Number of Queries Per Day (Last 30 Days)")
plt.xlabel("Date")
plt.ylabel("Number of Queries")
plt.xticks(rotation=45)
plt.tight_layout()
save_plot(fig5, "plots/daily_queries.png")

# 6. Feedback by the people 
positive_feedback = 0
negative_feedback = 0
positive_feedback_text = []
negative_feedback_text = []
values = []
labels = []

for i in range(feedback.shape[0]):
    if feedback['Feedback'][i]=='Good':
        positive_feedback +=1
        if not (pd.isna(feedback['Feedback Text'][i])):
            positive_feedback_text.append(feedback['Feedback Text'][i])
    elif feedback['Feedback'][i]=='Bad':
        negative_feedback +=1
        if not (pd.isna(feedback['Feedback Text'][i])):
            negative_feedback_text.append(feedback['Feedback Text'][i])
            
story.append(Paragraph(f"<b>Positive feedback Count:</b> {positive_feedback}", styles['EmojiText']))
if (positive_feedback>0):
    values.append(positive_feedback)
    labels.append('Positive Feedback')
    story.append(Paragraph(f"<b>Positive feedbacks:</b>", styles['EmojiText']))
    for text in positive_feedback_text:
        story.append(Paragraph(f"→ {text}", styles['EmojiText']))
story.append(Spacer(1, 12))

story.append(Paragraph(f"<b>Negative feedback Count:</b> {negative_feedback}", styles['EmojiText']))
if (negative_feedback>0):
    values.append(negative_feedback)
    labels.append('Negative Feedback')
    story.append(Paragraph(f"<b>Negative feedbacks:</b>", styles['EmojiText']))
    for text in negative_feedback_text:
        story.append(Paragraph(f"→ {text}", styles['EmojiText']))
story.append(Spacer(1, 12))

# Pie chart to show the feedback by people
fig6 = plt.figure(figsize=(6, 4))
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Distribution of Feedback")
plt.tight_layout()
save_plot(fig6, "plots/feedback_pie.png")

# Build PDF
pdf.build(story)
print("✅ Report saved as Chatbot_Report.pdf")
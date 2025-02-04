from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Set
import os
import random
import json
from urllib.parse import unquote
from threading import Thread
import re

app = Flask(__name__)
CORS(app)
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str
    explanation: str

# Global variables
question_cache: List[QuizQuestion] = []
used_questions: Set[str] = set()
current_topic: str = ""

def print_question(q: QuizQuestion, index: int):
    print(f"\nQuestion {index}:")
    print("=" * 50)
    print(f"Q: {q.question}")
    print("\nOptions:")
    for i, opt in enumerate(q.options):
        print(f"{chr(65+i)}) {opt}")
    print(f"\nCorrect Answer: {q.answer}")
    print(f"Explanation: {q.explanation}")
    print("-" * 50)

def read_chapter_content(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def calculate_accuracy(text_content: str, questions: list) -> float:
    try:
        total_words = len(text_content.split())
        relevant_count = 0
        for q in questions:
            question_words = q.question.lower().split()
            for word in question_words:
                if len(word) > 3 and word in text_content.lower():
                    relevant_count += 1
        accuracy = min((relevant_count / (len(questions) * 2)) * 100, 100)
        return round(accuracy, 2)
    except Exception as e:
        print(f"Error calculating accuracy: {str(e)}")
        return 0.0

def generate_quiz_questions(text_content: str = None, topic: str = None, is_practice_mode: bool = False) -> Optional[List[QuizQuestion]]:
    print("\nGenerating Questions...")
    print("=" * 50)

    system_prompt = """Generate thought-provoking multiple choice questions that enhance students' cognitive abilities and IQ. Include questions that:
    1. Test logical reasoning and pattern recognition
    2. Require application of concepts in novel situations
    3. Involve analysis and problem-solving
    4. Encourage creative thinking and innovation
    5. Integrate multiple concepts and ideas
    
    The response must be a JSON object with the following structure:
    {
        "questions": [
            {
                "question": "Question text",
                "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                "answer": "Correct option text",
                "explanation": "Brief explanation under 50 words"
            }
        ]
    }"""

    num_questions = 5

    if text_content:
        user_prompt = f"Content:\n{text_content}\n\nCreate {num_questions} questions in the specified JSON format."
    else:
        user_prompt = f"Create {num_questions} questions about {topic} in the specified JSON format."

    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content
        response_data = json.loads(response_text)
        
        processed_questions = []
        for q in response_data["questions"]:
            if not all(k in q for k in ["question", "options", "answer", "explanation"]):
                continue
            if len(q["options"]) != 4:
                continue
            if q["question"] in used_questions:
                continue

            question = QuizQuestion(
                question=q["question"],
                options=q["options"],
                answer=q["answer"],
                explanation=q["explanation"]
            )

            if question.answer not in question.options:
                continue

            random.shuffle(question.options)
            used_questions.add(question.question)
            processed_questions.append(question)

        for i, q in enumerate(processed_questions, 1):
            print_question(q, i)

        return processed_questions

    except Exception as e:
        print(f"Error in generate_quiz_questions: {str(e)}")
        return None

def preload_questions(standard: str, subject: str, chapter: str, topic: str, is_practice_mode: bool = False):
    global question_cache, current_topic, used_questions
    
    if topic != current_topic:
        question_cache.clear()
        used_questions.clear()
        current_topic = topic
    
    file_path = f"/home/ubuntu/schoolbooks/{standard}/{subject}/{topic}.txt"
    
    print(f"\nPreloading questions for {subject} Chapter {topic} (Standard {standard})")
    print("=" * 50)
    
    if os.path.exists(file_path):
        chapter_content = read_chapter_content(file_path)
        if chapter_content:
            questions = generate_quiz_questions(text_content=chapter_content, is_practice_mode=is_practice_mode)
            if questions:
                accuracy = calculate_accuracy(chapter_content, questions)
                print(f"\nQuestion Generation Accuracy: {accuracy}%")
                print("=" * 50)
                question_cache.extend(questions)
    else:
        print(f"\nFile not found: {file_path}")
        print("Generating questions based on topic instead...")
        questions = generate_quiz_questions(topic=topic, is_practice_mode=is_practice_mode)
        if questions:
            question_cache.extend(questions)

@app.route('/quiz/next', methods=['GET'])
def get_next_questions():
    try:
        topic = unquote(request.args.get('topic', ''))
        current_index = int(request.args.get('current_index', 0))
        standard = request.args.get('standard', '')
        subject = request.args.get('subject', '')
        chapter = request.args.get('chapter', '')
        is_practice_mode = request.args.get('is_practice_mode', 'false').lower() == 'true'
        
        if not topic:
            return jsonify({"error": "Missing topic parameter"}), 400

        topic = topic.strip()
        max_questions = 50 if is_practice_mode else 20
        
        if current_index >= max_questions:
            return jsonify({
                "questions": [],
                "should_fetch": False,
                "total_questions": max_questions
            })
        
        if current_index % 5 == 2 or len(question_cache) < 5:
            Thread(target=preload_questions, args=(standard, subject, chapter, topic, is_practice_mode)).start()

        if len(question_cache) < 5:
            if standard and subject and chapter:
                file_path = f"D:/bck/schoolbooks/{standard}th/{subject}/Chapter {chapter}.txt"
                if os.path.exists(file_path):
                    chapter_content = read_chapter_content(file_path)
                    questions = generate_quiz_questions(text_content=chapter_content, is_practice_mode=is_practice_mode)
                else:
                    questions = generate_quiz_questions(topic=topic, is_practice_mode=is_practice_mode)
            else:
                questions = generate_quiz_questions(topic=topic, is_practice_mode=is_practice_mode)
                
            if questions is None:
                return jsonify({"error": "Failed to generate questions"}), 500
        else:
            questions = question_cache[:5]
            del question_cache[:5]

        return jsonify({
            "questions": [q.model_dump() for q in questions],
            "should_fetch": True,
            "total_questions": max_questions
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/quiz/clear-cache', methods=['GET'])
def clear_cache():
    global question_cache, used_questions
    question_cache.clear()
    used_questions.clear()
    return jsonify({"status": "Cache cleared"}), 200

if __name__ == '__main__':
    print("\nStarting Quiz Generator Server...")
    print("=" * 50)
    CORS(app, resources={r"/": {"origins": ""}})
    app.run(debug=True)

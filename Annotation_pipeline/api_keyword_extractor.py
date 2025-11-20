from openai import OpenAI
import json

class GPTTextProcessor:
    def __init__(self, config_path="../config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.client = OpenAI(api_key=config["openai_api_key"])

    def process_text(self, result_with_timestamps):
        prompt = f"""
        You are an expert robotics data annotator. Your task is to analyze a sequence of transcribed words, each with a start and end timestamp, and segment them into two meaningful, high-level robotic action units "pick" and "place", and write down the corresponding start and end timestamp.
        Before segmenting, please revise the text I transcribed if there are any obvious problems. The background of the text is to pick up a blue cube from the table and place it into the red box.
        **CRITICAL RULES:**
        1.  **Action Unit:** An action unit (e.g., "pick") represents the **all meaningful phrases or sentences** describing that action, including the words of intent (e.g., "I want to...").
        2.  **Start Time:** The "start_time" **MUST** be the "start" timestamp of the **very first word** of the first phrase in that action unit (e.g., the timestamp for "I", not the timestamp for "pick").
        3.  **End Time:** The "end_time" **MUST** be the "end" timestamp of the **very last word** of the last phrase in that action unit (e.g., the timestamp for "finished").
        4.  **Grouping:** An action unit (like "pick") **includes all speech** until the *next* different high-level action (like "place") begins, or the task is declared "finished".
        Original transcribed text with timestamps:
        {result_with_timestamps}
        Output format:
        Corrected text: xxx
        List of action units with timestamps: [Pick: start timestamp - end timestamp, Place: start timestamp - end timestamp]
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant who is good at English text processing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, #output conservative answer
            max_tokens=500
        )
        return response.choices[0].message.content
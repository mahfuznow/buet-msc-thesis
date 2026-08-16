import json
import os
import textwrap

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Missing dependency: openai. Install with `pip install openai`.") from exc

root_env = "C:/Users/Shefayat/Desktop/Research/Thesis/banglahallueval/BanglaHalluEval/.env"
if load_dotenv is not None:
    load_dotenv(root_env)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key)

PROMPTS = {}
PROMPTS["PROMPT_1_FACTUAL"] = textwrap.dedent(
    """
    I want you act as a hallucination answer generator. The answer should be given in BANGLA. Given a question, right answer, and related knowledge, your objective is to write a hallucinated answer that sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the following method: You are trying to answer a question but there is a factual contradiction between the answer and the knowledge. You can fabricate some information that does not exist in the provided knowledge.
    #Knowledge#: "উইলিয়াম আব্রাহাম সাইমন ঔডারল্যান্ড (Dutch: Wiliam Ouderland) (জন্ম: ৬ ডিসেম্বর, ১৯১৭ — মৃত্যু: ১৮ই মে, ২০০১) ছিলেন একজন ওলন্দাজ-অস্ট্রেলীয় সামরিক কমান্ডো অফিসার। তিনি দ্বিতীয় বিশ্বযুদ্ধে সক্রিয়ভাবে অংশগ্রহণ করেন। বাংলাদেশ সরকার তাঁকে বাংলাদেশের চতুর্থ সর্বোচ্চ সামরিক খেতাব বীর প্রতীক প্রদান করে।"
    #Question#: "ওলন্দাজ-অস্ট্রেলীয় সামরিক কমান্ডো অফিসার উইলিয়াম আব্রাহাম সাইমন ঔডারল্যান্ড কবে জন্মগ্রহণ করেন ?"
    #Right Answer#: "৬ ডিসেম্বর, ১৯১৭"
    #Hallucinated Answer#: "৬ নভেম্বর, ১৯১৬"
    You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have about 5 more words than #Right Answer#.
    #Knowledge#: <insert the related knowledge/context>
    #Question#: <insert the question>
    #Right Answer#: <insert the right answer to the question>
    #Hallucinated Answer#: Generate
    """
).strip()

PROMPTS["PROMPT_2_COMPREHENSION"] = textwrap.dedent(
    """
    I want you act as a hallucination answer generator. The answer should be given in BANGLA. Given a question, right answer, and related knowledge, your objective is to write a hallucinated answer that sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the following method: You are trying to answer a question but you misunderstand the question context and intention.
    #Knowledge#: "1927-28 সালে ঢাকায় প্রথম চলচ্চিত্র নির্মিত হয়। নওয়াব পরিবারের কয়েকজন তরুণ সংস্কৃতিসেবী নির্মাণ করেন চলচ্চিত্র সুকুমারী।"
    #Question#: "স্বাধীন বাংলাদেশের প্রথম চলচ্চিত্রটির নাম কী ?"
    #Right Answer#: "সুকুমারী"
    #Hallucinated Answer#: "জাহির রাইহান"
    You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have about 5 more words than #Right Answer#.
    #Knowledge#: <insert the related knowledge/context>
    #Question#: <insert the question>
    #Right Answer#: <insert the right answer to the question>
    #Hallucinated Answer#: Generate
    """
).strip()

PROMPTS["PROMPT_3_SPECIFICITY"] = textwrap.dedent(
    """
    I want you act as a hallucination answer generator. The answer should be given in BANGLA. Given a question, right answer, and related knowledge, your objective is to write a hallucinated answer that sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the following method: You are trying to answer a question but the answer is too general or too specific to answer the question at an appropriate level of specificity.
    #Knowledge#: "খুলনা প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয় (কুয়েট) বাংলাদেশের একটি অন্যতম সরকারি প্রকৌশল বিশ্ববিদ্যালয়। এখানে প্রায় ৬ হাজার জন ছাত্রছাত্রী পড়াশোনা করছে।"
    #Question#: "বর্তমানে খুলনা প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয়ের মোট ছাত্রছাত্রীর সংখ্যা কত ?"
    #Right Answer#: "প্রায় ৬ হাজার"
    #Hallucinated Answer#: "অজানা"
    You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have about 5 more words than #Right Answer#.
    #Knowledge#: <insert the related knowledge/context>
    #Question#: <insert the question>
    #Right Answer#: <insert the right answer to the question>
    #Hallucinated Answer#: Generate
    """
).strip()

PROMPTS["PROMPT_4_INFERENCE"] = textwrap.dedent(
    """
    I want you act as a hallucination answer generator. The answer should be given in BANGLA. Given a question, right answer, and related knowledge, your objective is to write a hallucinated answer that sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the following method: You are trying to answer a question but the answer cannot be inferred from the knowledge. You can incorrectly reason with the knowledge to arrive at a hallucinated answer.
    #Knowledge#: "ঢাকা দক্ষিণ এশিয়ার রাষ্ট্র বাংলাদেশের রাজধানী ও বৃহত্তম শহর। ভৌগোলিকভাবে এটি বাংলাদেশের মধ্যভাগে বুড়িগঙ্গা নদীর উত্তর তীরে অবস্থিত।"
    #Question#: "ঢাকার মোট আয়তন কত ?"
    #Right Answer#: "১৩৪ বর্গমাইল"
    #Hallucinated Answer#: "২০ মিলিয়ন জনসংখ্যা"
    You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have about 5 more words than #Right Answer#.
    #Knowledge#: <insert the related knowledge/context>
    #Question#: <insert the question>
    #Right Answer#: <insert the right answer to the question>
    #Hallucinated Answer#: Generate
    """
).strip()

SYSTEM = (
    "Convert only the Bangla text into Banglish (Latin-script Bengali). "
    "Keep English text unchanged. Preserve punctuation, placeholders, and formatting. "
    "Do not translate labels like #Knowledge#, #Question#, #Right Answer#, #Hallucinated Answer#, "
    "or the word Generate. Keep numbers as-is. Output exactly the converted prompt text, "
    "no extra commentary."
)

out = {}
for key, prompt in PROMPTS.items():
    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=1200,
        temperature=0,
    )
    out[key] = response.output_text

print(json.dumps(out, ensure_ascii=False, indent=2))

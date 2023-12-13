import os
import openai
from datasets import load_dataset


def story_preprocess_function(examples):
    """This preprocess function combines all the sentences together and outputs the story that is plausible"""
    story1 = [" ".join(stories[0]["sentences"]) for stories in examples["stories"]]
    story2 = [" ".join(stories[1]["sentences"]) for stories in examples["stories"]]
    story_labels = examples["label"]
    confl_pairs = examples["confl_pairs"]
    return story1, story2, story_labels, confl_pairs


def gpt(story1, story2, label, pair):
    try:
        processed_story1 = processStory(story1)
        processed_story2 = processStory(story2)

        prompt = f"""You are an expert on logic. You will be given two stories.
    You have two tasks:
    1. determine which story is implausible (0 or 1)
    2. determine which two sentences in the implausible story conflicts.

    The output should be 3 numbers split by space. The first 2 numbers should be the conflict sentences pair (0-index). The third number should be the implausible story.  Do not add any other explanations.

    Example1:

    # Question:

    Story 0:
    0. Ann sat in the chair.
    1. Ann unplugged the telephone.
    2. Ann picked up a pencil.
    3. Ann opened the book.
    4. Ann wrote in the book.

    Story 1:
    0. Ann sat in the chair.
    1. Ann unplugged the telephone.
    2. Ann picked up a pencil.
    3. Ann opened the book.
    4. Ann heard the telephone ring.

    # Answer:

    1 4 1

    Example2:

    # Question:

    Story 0:
    0. John was getting the snacks ready for the party.
    1. John opened the cabinet, took out a pan and put it on the counter.
    2. John opened the fridge and got out the pizza.
    3. The pizza smelled funny so John threw it in the trash.
    4. John took a knife and cut the hot pizza in eight slices.

    Story 1:
    0. John was getting the snacks ready for the party.
    1. John opened the cabinet, took out a pan and put it on the counter.
    2. John opened the fridge and got out the pizza.
    3. John put the pizza on the pan and put them into the oven.
    4. John took a knife and cut the hot pizza in eight slices.

    # Answer:

    3 4 0

    # Question:

    Story 0:
    {processed_story1}
    Story 1:
    {processed_story2}

    # Answer:

    """

        openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        assert openai_api_key is not None, "OpenAI API key not found."

        client = openai.OpenAI(
            # This is the default and can be omitted
            api_key=openai_api_key
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        chat_completion = client.chat.completions.create(
            messages=messages, model="gpt-4", temperature=0
        )

        result = chat_completion.choices[0].message.content
        # print(result)
        tmp = result.split()
        accuracy_flag = True
        consistency_flag = True
        pred_pairs = [int(tmp[0]), int(tmp[1])]
        pred_label = int(tmp[2])
        if pred_label == label:
            accuracy_flag = False
        if pred_pairs not in pair:
            consistency_flag = False
        return accuracy_flag, consistency_flag
    except Exception as e:
        return False, False


def processStory(story):
    tmp = story.split(".")[:-1]
    result = ""
    for i, sen in enumerate(tmp):
        result += str(i) + ". " + sen.strip() + "\n"
    return result


def main():
    dataset = load_dataset("sled-umich/TRIP")
    dataset = dataset["ClozeTest"]
    story1, story2, labels, pairs = story_preprocess_function(dataset)
    n = len(story1)

    accuracy_cnt = 0
    consistency_cnt = 0

    for i in range(n):
        if i % 30 == 0:
            print("Start Query ", i)
        accuracy_flag, consistency_flag = gpt(story1[i], story2[i], labels[i], pairs[i])
        if accuracy_flag:
            accuracy_cnt += 1
        if consistency_flag:
            consistency_cnt += 1

    accuracy = accuracy_cnt / n
    consistency = consistency_cnt / n

    print("Accuracy: ", accuracy)
    print("Consistency", consistency)


if __name__ == "__main__":
    main()

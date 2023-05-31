from src.models.simple_index import simple_index

if __name__ == "__main__":
    document = """
    Once upon a time, in the small town of Willowbrook, there lived a curious young girl named Emily. 
    With her bright eyes and an insatiable thirst for adventure, she was always seeking new experiences and discoveries.
    One sunny morning, Emily ventured into the dense forest that surrounded her town. 
    She had heard tales of a hidden waterfall deep within the woods, and her heart was set on finding it. 
    Armed with her trusty backpack and a sense of determination, she followed a faint trail leading her deeper into the wilderness.
    As she walked, Emily noticed the forest changing around her. The air grew cooler, 
    and beams of sunlight filtered through the thick canopy of leaves, creating a magical ambiance. 
    She could hear the soft chirping of birds and the rustling of leaves under her feet. 
    It felt as though nature itself was guiding her towards her destination.
    After hours of hiking, Emily's perseverance paid off. She stumbled upon a hidden clearing bathed in golden sunlight. 
    In the center stood a majestic waterfall, 
    cascading down from the rocks above into a crystal-clear pool below. The sight was breathtaking, 
    and Emily couldn't help but be in awe of its beauty.
    """
    question = "where did emily live ?"

    index = simple_index()
    answer = index(context=document, question=question)

    print(answer)

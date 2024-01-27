from setuptools import setup, find_packages

def get_requirements(filepath: str):
    """
    Read a file containing requirements and return a list of filtered requirements.

    Parameters:
    - filepath (str): The path to the file containing requirements.

    Returns:
    - list: A list of filtered requirements where items equal to CONST_HYPHEN_DOT_E are excluded.
    """

    CONST_HYPHEN_DOT_E = '-e .'
    requirements_list = []

    with open(filepath) as file_obj:
        requirements_list = file_obj.readlines()

    # Use a list comprehension to filter out items equal to CONST_HYPHEN_DOT_E
    requirements_list = [requirements.lower().strip().replace('\n', '') for requirements in requirements_list if requirements.lower().strip() != CONST_HYPHEN_DOT_E]

    return requirements_list

  
setup( 
    name='Quora question pair similarity', 
    version='0.1',  
    author='sachin kapoor', 
    author_email='sachinkapoor2055@gmail.com', 
    packages=find_packages(), 
    install_requires=get_requirements('requirements.txt'), 
) 
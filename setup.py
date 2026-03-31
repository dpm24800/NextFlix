from setuptools import setup, find_packages

# def get_requirements(path):
#     with open(path, 'r') as f:
#         return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def get_requirements(path):
    """
    Read requirements.txt and return a list of dependencies.
    Skips empty lines and comments.

    Reads each line
    Removes spaces
    Skips empty lines and comments
    Stores valid lines in a list
    """
    requirements = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "": continue # Skip empty lines            
            if line.startswith('#'): continue # Skip comments
            requirements.append(line)
    
    return requirements

setup(
    name="NextFlix",
    version="1.0.0",
    description="An end-to-end content-based movie recommendation system using TF-IDF vectorization and cosine similarity",
    author_email='dpm.it24800@gmail.com',
    author="Dipak Pulami Magar",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
from setuptools import setup, find_packages

def get_requirements(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="NextFlix",
    version="1.0.0",
    description="An end-to-end content-based movie recommendation system using TF-IDF",
    author_email='dpm.it24800@gmail.com',
    author="Dipak Pulami Magar",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

from setuptools import setup, find_packages

setup(
    name="robot_dynamics",
    version="0.1",
    packages=find_packages(include=[
        'controllers',
        'dynamic_modelling_methods',
        'plotting',
        'robot_model',
        # We don't include 'examples', 'data', 'material' as they're not Python packages
    ]),
    package_data={
        # Include any .pkl files in the models directory
        'models': ['*.pkl'],
        # Include any data files
        'data': ['*.npz'],
    },
    install_requires=[
        'numpy',
        'sympy',
        'matplotlib',
        'dill',
        'scipy',
        # Add other dependencies your project needs
    ],
    author="Akash Vyas",
    description="A package for robot dynamics and control",
    python_requires='>=3.8'
)
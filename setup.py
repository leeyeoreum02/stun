import os
import io
from setuptools import setup, find_packages


def get_install_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith('#')]
    return reqs


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
        return f.read()


if __name__ == '__main__':
    setup(
        name='stun',
        version='0.0.0',
        description='Swin Triple Upsampling Network (STUN)',
        # long_description=get_long_description(),
        # long_description_content_type='text/markdown',
        author='Summer Lee, Una Yeo',
        python_requires='<3.10',
        install_requires=get_install_requirements(),
        classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
        ],
        packages=find_packages(),
        dependency_links=[
            'https://download.pytorch.org/whl/torch_stable.html',
        ],
    )

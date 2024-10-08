from setuptools import setup, find_packages

NAME = 'BinaryJax'

if __name__ == '__main__':
    setup(
        name=NAME,
        author='Haibin Ren',
        author_email='rhb23@mails.tsinghua.edu.cn',
        version='0.1',
        description='A JAX-based package for binary microlensing',
        license='MIT',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/CoastEgo/BinaryJax',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
        ],
        install_requires=[
            'jax>=0.4.13',
            'jaxlib>=0.4.13',
            'numpy',
            'scipy',
            'matplotlib'
        ],
        python_requires='>=3.8',
    )
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="askgloom",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A browser automation tool with Telegram integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ask-gloom",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "selenium>=4.0.0",
        "python-telegram-bot>=13.0",
        "flask>=2.0.0",
        "webdriver_manager>=3.5.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
    ],
    entry_points={
        "console_scripts": [
            "askgloom=askgloom.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "askgloom": [
            "ui/static/*",
            "ui/static/styles/*",
            "ui/templates/*",
        ],
    },
)
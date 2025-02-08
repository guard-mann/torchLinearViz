from setuptools import setup, find_packages, Extension
import pybind11
import torch
import sys
import os




setup(
    name='torchlinearviz',  # モジュール名（PyPIで公開される名前）
    version='0.1.0',  # 初期バージョン
    description='A tool to visualize PyTorch model graphs in the browser.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # README形式
    author='guard-mann',
    author_email='hoge@gmail.com',
    url='https://github.com/guard-mann/torchLinearViz',  # GitHubリポジトリのURL
    packages=find_packages(),  # `graphvisualizer`以下を自動的にパッケージ化
    include_package_data=True,  # 静的ファイルやリソースも含む
    install_requires=[
        'flask',
        'flask-socketio',
        'torch',  # PyTorchを依存パッケージに追加
        'pybind11',
    ],
    entry_points={
        'console_scripts': [
            'torchlinearviz=torchLinearViz.visualization.serverSocket:main',  # 実行可能コマンドを登録
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Pythonのバージョン制約
)

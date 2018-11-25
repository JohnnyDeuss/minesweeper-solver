from setuptools import setup, find_packages


setup(name='minesweeper_solver',
      version='1.0',
      description='Minesweeper solver',
      author='Johnny Deuss',
      author_email='johnnydeuss@gmail.com',
      url='https://github.com/JohnnyDeuss/minesweeper_solver',
      project_urls={'Source': 'https://github.com/JohnnyDeuss/minesweeper_solver'},
      install_requires=['scipy', 'numpy', 'python-constraint'],
      extras_require={
          'GUI': ['minesweeper']
      },
      dependency_links=[
          'git+ssh://git@github.com/JohnnyDeuss/minesweeper#egg=minesweeper',
      ],
      packages=find_packages()
    )

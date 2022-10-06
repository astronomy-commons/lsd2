from setuptools import setup

setup(
    name="hipscat-config",
    author="Sam Wyatt",
    url="https://github.com/astronomy-commons/lsd2",
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    packages=["hipscat"],
    include_package_data=True
)

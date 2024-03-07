import os
import importlib
import gops.create_pkg.create_env as ce
from gops.env.tools.env_data_checker import check_env, simple_check_env
from gops.env.tools.env_model_checker import check_model
from gops.env.tools.get_all_envs import get_env_model_files, CLASSIC, TOY_TEXT, BOX2D, MUJOCO


def simple_check_on_windows():
    """
    check all env (try to make each env), mujoco env not included on windows
    """

    for env_name in CLASSIC + TOY_TEXT + BOX2D:
        simple_check_env(env_name)


def simple_check_on_linux():
    """
    check all env, mujoco included
    """
    for env_name in CLASSIC + TOY_TEXT + BOX2D + MUJOCO:
        simple_check_env(env_name)


def main():
    env_list, model_list = get_env_model_files()
    for e in env_list:
        check_env(e[:-5])

    for m in model_list:
        check_model(m[:-6])


if __name__ == '__main__':
    # main()
    simple_check_on_windows()
    # simple_check_on_linux()
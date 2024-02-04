import hashlib


def get_exp_group(user_id: int) -> str:
    out = "control"
    group_ind = (int(hashlib.md5((str(user_id) + "final_project_salt").encode()).hexdigest(), 16) % 2)
    if group_ind:
        out = "test"
    return out

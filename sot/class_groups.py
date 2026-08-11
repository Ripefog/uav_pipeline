"""Nhóm class cho guard — map theo TÊN, không theo id.

Hardcode id class là SAI ÂM THẦM khi đổi weight sang taxonomy khác (id 3 của
VisDrone là 'car', của COCO là 'motorcycle'). Bảng này viết theo tên; id được
resolve lúc chạy qua detector.names (weights/names.yaml).
"""
from typing import Dict, List, Optional

# gate=class: nhóm hẹp. VisDrone tách người thành pedestrian (đang đi/đứng) và
# people (tư thế khác) — cùng một "person" nên gộp.
GROUP_BY_NAME: Dict[str, List[str]] = {
    "pedestrian": ["pedestrian", "people"],
    "people": ["pedestrian", "people"],
    "bicycle": ["bicycle"],
    "car": ["car"],
    "van": ["van"],
    "truck": ["truck"],
    "tricycle": ["tricycle"],
    "awning-tricycle": ["awning-tricycle"],
    "bus": ["bus"],
    "motor": ["motor"],
}

# gate=family: nới ra vì detector lẫn class nặng. Đo tại vị trí GT bus của
# uav0000305_00000_v: detector gán van 27 frame, bus 12, car 8, không có gì 17
# frame (miss rate 78%). Gate hẹp ở đây sẽ cắt oan.
FAMILY_BY_NAME: Dict[str, List[str]] = {
    "pedestrian": ["pedestrian", "people"],
    "people": ["pedestrian", "people"],
    "car": ["car", "van", "truck", "bus"],
    "van": ["car", "van", "truck", "bus"],
    "truck": ["car", "van", "truck", "bus"],
    "bus": ["car", "van", "truck", "bus"],
    "bicycle": ["bicycle", "tricycle", "awning-tricycle", "motor"],
    "tricycle": ["bicycle", "tricycle", "awning-tricycle", "motor"],
    "awning-tricycle": ["bicycle", "tricycle", "awning-tricycle", "motor"],
    "motor": ["bicycle", "tricycle", "awning-tricycle", "motor"],
}


def accepted_ids(gate: str, init_cls: int,
                 names: Dict[int, str]) -> Optional[List[int]]:
    """Id class được coi là 'cùng nhóm' với class lúc init.

    Trả None = BỎ QUA điều kiện class. Xảy ra khi gate='presence', hoặc khi tên
    class không có trong bảng — trả None (chứ không phải list rỗng) là cố ý: list
    rỗng sẽ chặn mọi detection và làm guard cắt oan mọi lúc.
    """
    if gate == "presence":
        return None
    table = FAMILY_BY_NAME if gate == "family" else GROUP_BY_NAME
    init_name = names.get(int(init_cls), "")
    group = table.get(init_name)
    if group is None:
        print(f"[sot/guard] class '{init_name}' (id {init_cls}) không có trong bảng "
              f"nhóm -> hạ gate '{gate}' xuống 'presence'")
        return None
    name_to_id = {v: k for k, v in names.items()}
    return sorted(name_to_id[n] for n in group if n in name_to_id)

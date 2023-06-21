import numpy as np
from scipy.spatial.transform import Rotation as R


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    def get_joint_rotations():
        joint_rotations = np.empty(joint_orientations.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1 or joint_name[i].endswith('_end'):
                joint_rotations[i] = joint_orientations[i]
            else:
                joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat()
        return joint_rotations

    def get_joint_offsets():
        joint_offsets = np.empty(joint_positions.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_offsets[i] = np.array([0.,0.,0.])
            else:
                joint_offsets[i] = joint_initial_position[i] - joint_initial_position[joint_parent[i]]
        return joint_offsets

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint

    # 每个joint的local rotation，用四元数表示
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    rotation_chain = np.empty((len(path), 4))
    position_chain = np.empty((len(path), 3))
    
    for i in range(len(path)):
        index = path[i]
        position_chain[i] = joint_positions[index]
        if index in path2 and joint_parent[index] != -1:
            rotation_chain[i] = R.from_quat(joint_rotations[joint_parent[index]]).inv().as_quat()
        else:
            rotation_chain[i] = joint_rotations[index]
    
    # CCD IK
    times = 1
    distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))
    while times > 0 and distance > 0.001:
        print('-' * 100)
        print(distance)
        times -= 1
        for i in range(len(path) - 2, 0, -1):
            cur_pos = position_chain[i]
            # 计算旋转的轴角表示
            c2t = target_pose - cur_pos
            c2e = position_chain[-1] - cur_pos
            axis = np.cross(c2e, c2t)
            axis = axis / np.linalg.norm(axis)
            theta = np.arccos(np.dot(c2e, c2t) / (np.linalg.norm(c2e) * np.linalg.norm(c2t)))
            print(theta)
            delta_rotation = R.from_rotvec(theta * axis)
            # 更新当前的local rotation 和子关节的position, orientation
            rotation_chain[i] = (delta_rotation * R.from_quat(rotation_chain[i])).as_quat()
            joint_orientations[path[i]] = (delta_rotation * R.from_quat(joint_orientations[path[i]])).as_quat()
            for j in range(i + 1, len(path)):
                # joint_orientations[path[j]] = (delta_rotation * R.from_quat(joint_orientations[path[j]])).as_quat()
                joint_orientations[path[j]] = (R.from_quat(joint_orientations[path[j - 1]]) * R.from_quat(rotation_chain[j])).as_quat()
                position_chain[j] = np.dot(R.from_quat(joint_orientations[path[j - 1]]).as_matrix(), joint_offsets[path[j]]) + position_chain[j - 1]
            distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))

    
    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        joint_positions[index] = position_chain[i]
        if index in path2 and joint_parent[index] != -1:
            joint_rotations[joint_parent[index]] = R.from_quat(rotation_chain[i]).inv().as_quat()
        else:
            joint_rotations[index] = rotation_chain[i]

    # return joint_positions, joint_orientations

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if joint_parent.index(-1) in path:
        root_index = path.index(joint_parent.index(-1))
        if root_index != 0:
            root_pos = position_chain[root_index]
            root_orientation = rotation_chain[0]
            for i in range(1, root_index + 1):
                root_orientation = root_orientation * rotation_chain[i]
            joint_orientations[0] = root_orientation
            joint_positions[0] = root_pos
    
    # 最后计算一遍FK，得到更新后的position和orientation
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])
        
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations
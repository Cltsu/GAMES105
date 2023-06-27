import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
import task2_inverse_kinematics

# CCD 循环坐标下降
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
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
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

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #
    if len(path2) == 1 and path2[0] != 0:
        path2 = []

    # 每个joint的local rotation，用四元数表示
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()


    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    rotation_chain = np.empty((len(path),), dtype=object)
    position_chain = np.empty((len(path), 3))
    orientation_chain = np.empty((len(path),), dtype=object)
    offset_chain = np.empty((len(path), 3))

    # 对chain进行初始化
    if len(path2) > 1:
        orientation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv()
    else:
        orientation_chain[0] = R.from_quat(joint_orientations[path[0]])

    position_chain[0] = joint_positions[path[0]]
    rotation_chain[0] = orientation_chain[0]
    offset_chain[0] = np.array([0.,0.,0.])

    for i in range(1, len(path)):
        index = path[i]
        position_chain[i] = joint_positions[index]
        if index in path2:
            # essential
            orientation_chain[i] = R.from_quat(joint_orientations[path[i + 1]])
            rotation_chain[i] = R.from_quat(joint_rotations[path[i]]).inv()
            offset_chain[i] = -joint_offsets[path[i - 1]]
            # essential
        else:
            orientation_chain[i] = R.from_quat(joint_orientations[index])
            rotation_chain[i] = R.from_quat(joint_rotations[index])
            offset_chain[i] = joint_offsets[index]


    # CCD IK
    times = 10
    distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))
    end = False
    while times > 0 and distance > 0.001 and not end:
        times -= 1
        # 先动手
        for i in range(len(path) - 2, -1, -1):
        # 先动腰
        # for i in range(1, len(path) - 1):
            if joint_parent[path[i]] == -1:
                continue
            cur_pos = position_chain[i]
            # 计算旋转的轴角表示
            c2t = target_pose - cur_pos
            c2e = position_chain[-1] - cur_pos
            axis = np.cross(c2e, c2t)
            axis = axis / np.linalg.norm(axis)
            # 由于float的精度问题，cos可能cos(theta)可能大于1.
            cos = min(np.dot(c2e, c2t) / (np.linalg.norm(c2e) * np.linalg.norm(c2t)), 1.0)
            theta = np.arccos(cos)
            # 防止quat为0？
            if theta < 0.0001:
                continue
            delta_rotation = R.from_rotvec(theta * axis)
            # 更新当前的local rotation 和子关节的position, orientation
            orientation_chain[i] = delta_rotation * orientation_chain[i]
            rotation_chain[i] = orientation_chain[i - 1].inv() * orientation_chain[i]
            for j in range(i + 1, len(path)):
                orientation_chain[j] = orientation_chain[j - 1] * rotation_chain[j]
                position_chain[j] = np.dot(orientation_chain[j - 1].as_matrix(), offset_chain[j]) + position_chain[j - 1]
            distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))


    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        joint_positions[index] = position_chain[i]
        if index in path2:
            joint_rotations[index] = rotation_chain[i].inv().as_quat()
        else:
            joint_rotations[index] = rotation_chain[i].as_quat()

    if path2 == []:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() * orientation_chain[0]).as_quat()

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if joint_parent.index(-1) in path:
        root_index = path.index(joint_parent.index(-1))
        if root_index != 0:
            joint_orientations[0] = orientation_chain[root_index].as_quat()
            joint_positions[0] = position_chain[root_index]


    # 最后计算一遍FK，得到更新后的position和orientation
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])


    return joint_positions, joint_orientations



# 使用pytorch的自动微分进行梯度下降优化
def part1_inverse_kinematics_torch(meta_data, joint_positions, joint_orientations, target_pose):
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
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
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

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #
    if len(path2) == 1:
        path2 = []

    # 每个joint的local rotation，用四元数表示
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()


    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    rotation_chain = np.empty((len(path), 3), dtype=float)
    offset_chain = np.empty((len(path), 3), dtype=float)

    # 对chain进行初始化
    if len(path2) > 1:
        rotation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv().as_euler('XYZ')
    else:
        rotation_chain[0] = R.from_quat(joint_orientations[path[0]]).as_euler('XYZ')

    # position_chain[0] = joint_positions[path[0]]
    start_position = torch.tensor(joint_positions[path[0]], requires_grad=False)
    offset_chain[0] = np.array([0.,0.,0.])

    for i in range(1, len(path)):
        index = path[i]
        if index in path2:
            # essential
            rotation_chain[i] = R.from_quat(joint_rotations[path[i]]).inv().as_euler('XYZ')
            offset_chain[i] = -joint_offsets[path[i - 1]]
            # essential
        else:
            rotation_chain[i] = R.from_quat(joint_rotations[index]).as_euler('XYZ')
            offset_chain[i] = joint_offsets[index]

    # pytorch autograde
    rotation_chain_tensor = torch.tensor(rotation_chain, requires_grad=True, dtype=torch.float32)
    offset_chain_tensor = torch.tensor(offset_chain, requires_grad=False, dtype=torch.float32)
    target_position = torch.tensor(target_pose, requires_grad=False, dtype=torch.float32)
    rootjoint_index_in_path = path.index(0)
    max_times = 50
    lr = 0.1
    while max_times > 0:
        # 向前计算end position
        max_times -= 1
        cur_position = start_position
        cur_orientation = rotation_chain_tensor[0]
        for i in range(1, len(path)):
            cur_position = euler_angles_to_matrix(cur_orientation, 'XYZ') @ offset_chain_tensor[i] + cur_position
            orientation_matrix = euler_angles_to_matrix(cur_orientation, 'XYZ') @ euler_angles_to_matrix(rotation_chain_tensor[i], 'XYZ')
            cur_orientation = matrix_to_euler_angles(orientation_matrix, 'XYZ')
            # joint_positions[path[i]] = cur_position.detach().numpy()
            # joint_orientations[path[i]] = R.from_euler('XYZ', cur_orientation.detach().numpy()).as_quat()
        dist = torch.norm(cur_position - target_position)
        if dist < 0.01 or max_times == 0:
            break

        # 反向传播
        dist.backward()
        rotation_chain_tensor.grad[rootjoint_index_in_path].zero_()
        rotation_chain_tensor.data.sub_(rotation_chain_tensor.grad * lr)
        rotation_chain_tensor.grad.zero_()

    # return joint_positions, joint_orientations

    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        if index in path2:
            joint_rotations[index] = R.from_euler('XYZ', rotation_chain_tensor[i].detach().numpy()).inv().as_quat()
        else:
            joint_rotations[index] = R.from_euler('XYZ', rotation_chain_tensor[i].detach().numpy()).as_quat()


    # 当IK链不过rootjoint时，IK起点的rotation需要特殊处理
    if path2 == [] and path[0] != 0:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() 
                                    * R.from_euler('XYZ', rotation_chain_tensor[0].detach().numpy())).as_quat()

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if 0 in path and rootjoint_index_in_path != 0:
        rootjoint_pos = start_position
        rootjoint_ori = rotation_chain_tensor[0]
        for i in range(1, rootjoint_index_in_path + 1):
            rootjoint_pos = euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ offset_chain_tensor[i] + rootjoint_pos
            rootjoint_ori = matrix_to_euler_angles(euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ euler_angles_to_matrix(rotation_chain_tensor[i], 'XYZ'), 'XYZ')
        joint_orientations[0] = R.from_euler('XYZ', rootjoint_ori.detach().numpy()).as_quat()
        joint_positions[0] = rootjoint_pos.detach().numpy()

    
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
    target_pose = np.array([relative_x + joint_positions[0][0], target_height, relative_z + joint_positions[0][2]])
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)
    return joint_positions, joint_orientations



def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    def get_joint_rotations():
        joint_rotations = np.empty(joint_orientations.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
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

    metadata_right = task2_inverse_kinematics.MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'rWrist_end')

    lpath, path_name, lpath1, lpath2 = meta_data.get_path_from_root_to_end()
    rpath, path_name, rpath1, rpath2 = metadata_right.get_path_from_root_to_end()

    common_ancestor = 2
    # path1 里面没有rootjoint
    rpath1 = list(reversed(rpath1))[common_ancestor:]

    if len(lpath2) == 1:
        lpath2 = []
    if len(rpath2) == 1:
        rpath2 = []

    # 每个joint的local rotation，用四元数表示
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()


    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    lrotation_chain = np.empty((len(lpath), 3), dtype=float)
    loffset_chain = np.empty((len(lpath), 3), dtype=float)

    # 对chain进行初始化
    if len(lpath2) > 1:
        lrotation_chain[0] = R.from_quat(joint_orientations[lpath2[1]]).inv().as_euler('XYZ')
    else:
        lrotation_chain[0] = R.from_quat(joint_orientations[lpath[0]]).as_euler('XYZ')

    loffset_chain[0] = np.array([0.,0.,0.])
    start_position = torch.tensor(joint_positions[lpath[0]], requires_grad=False)
    
    for i in range(1, len(lpath)):
        index = lpath[i]
        if index in lpath2:
            # essential
            lrotation_chain[i] = R.from_quat(joint_rotations[lpath[i]]).inv().as_euler('XYZ')
            loffset_chain[i] = -joint_offsets[lpath[i - 1]]
            # essential
        else:
            lrotation_chain[i] = R.from_quat(joint_rotations[index]).as_euler('XYZ')
            loffset_chain[i] = joint_offsets[index]


    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    rrotation_chain = np.empty((len(rpath1), 3), dtype=float)
    roffset_chain = np.empty((len(rpath1), 3), dtype=float)

    for i in range(len(rpath1)):
        index = rpath1[i]
        rrotation_chain[i] = R.from_quat(joint_rotations[index]).as_euler('XYZ')
        roffset_chain[i] = joint_offsets[index]

    ########################

    # pytorch autograde
    lrotation_chain_tensor = torch.tensor(lrotation_chain, requires_grad=True, dtype=torch.float32)
    loffset_chain_tensor = torch.tensor(loffset_chain, requires_grad=False, dtype=torch.float32)
    rootjoint_index_in_lpath = lpath.index(0)

    rrotation_chain_tensor = torch.tensor(rrotation_chain, requires_grad=True, dtype=torch.float32)
    roffset_chain_tensor = torch.tensor(roffset_chain, requires_grad=False, dtype=torch.float32)
    rootjoint_index_in_rpath = rpath.index(0)

    left_target_position = torch.tensor(left_target_pose, requires_grad=False)
    right_target_position = torch.tensor(right_target_pose, requires_grad=False)

    max_times = 1000
    lr = 0.01
    common_ancestor = 2
    while max_times > 0:
        # 向前计算end position

        # compute left dist
        max_times -= 1
        cur_position = start_position
        cur_orientation = lrotation_chain_tensor[0]
        for i in range(1, len(lpath)):
            cur_position = euler_angles_to_matrix(cur_orientation, 'XYZ') @ loffset_chain_tensor[i] + cur_position
            orientation_matrix = euler_angles_to_matrix(cur_orientation, 'XYZ') @ euler_angles_to_matrix(lrotation_chain_tensor[i], 'XYZ')
            cur_orientation = matrix_to_euler_angles(orientation_matrix, 'XYZ')
            if lpath[i] == common_ancestor:
                ca_orientation = cur_orientation.clone()
                ca_position = cur_position.clone()
        ldist = torch.norm(cur_position - left_target_position)

        # compute right dist
        rcur_orientation = ca_orientation
        rcur_position = ca_position
        for i in range(len(rpath1)):
            rcur_position = euler_angles_to_matrix(rcur_orientation, 'XYZ') @ roffset_chain_tensor[i] + rcur_position
            rorientation_matrix = euler_angles_to_matrix(rcur_orientation, 'XYZ') @ euler_angles_to_matrix(rrotation_chain_tensor[i], 'XYZ')
            rcur_orientation = matrix_to_euler_angles(rorientation_matrix, 'XYZ')
        rdist = torch.norm(rcur_position - right_target_position)

        dist = ldist + rdist
        if dist < 0.01 or max_times == 0:
            break

        # 反向传播
        dist.backward()
        lrotation_chain_tensor.grad[rootjoint_index_in_lpath].zero_()
        lrotation_chain_tensor.data.sub_(lrotation_chain_tensor.grad * lr)
        lrotation_chain_tensor.grad.zero_()

        rrotation_chain_tensor.data.sub_(rrotation_chain_tensor.grad * lr)
        rrotation_chain_tensor.grad.zero_()

    # return joint_positions, joint_orientations

    # 把left链条的旋转写回joint_rotation
    for i in range(len(lpath)):
        index = lpath[i]
        if index in lpath2:
            joint_rotations[index] = R.from_euler('XYZ', lrotation_chain_tensor[i].detach().numpy()).inv().as_quat()
        else:
            joint_rotations[index] = R.from_euler('XYZ', lrotation_chain_tensor[i].detach().numpy()).as_quat()


    # 把right链条的旋转写回joint_rotation
    for i in range(len(rpath1)):
        joint_rotations[rpath1[i]] = R.from_euler('XYZ', rrotation_chain_tensor[i].detach().numpy()).as_quat()


    # 当IK链不过rootjoint时，IK起点的rotation需要特殊处理
    if lpath2 == [] and lpath[0] != 0:
        joint_rotations[lpath[0]] = (R.from_quat(joint_orientations[joint_parent[lpath[0]]]).inv() 
                                    * R.from_euler('XYZ', lrotation_chain_tensor[0].detach().numpy())).as_quat()

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if 0 in lpath and rootjoint_index_in_lpath != 0:
        rootjoint_pos = start_position
        rootjoint_ori = lrotation_chain_tensor[0]
        for i in range(1, rootjoint_index_in_lpath + 1):
            rootjoint_pos = euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ loffset_chain_tensor[i] + rootjoint_pos
            rootjoint_ori = matrix_to_euler_angles(euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ euler_angles_to_matrix(lrotation_chain_tensor[i], 'XYZ'), 'XYZ')
        joint_orientations[0] = R.from_euler('XYZ', rootjoint_ori.detach().numpy()).as_quat()
        joint_positions[0] = rootjoint_pos.detach().numpy()

    
    # 最后计算一遍FK，得到更新后的position和orientation
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])

    return joint_positions, joint_orientations




















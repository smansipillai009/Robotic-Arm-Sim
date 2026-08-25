"""
FK/IK Validator — 2-link, 3-DOF robotic arm
=============================================
Validates the forward/inverse kinematics before porting to the microcontroller.
Convention matches kinematics/joint-convention.md:
  - theta1: base yaw (about vertical Z), 0deg = pointing along +X
  - theta2: shoulder pitch from horizontal, math angle (absolute)
  - theta3_abs: elbow, math angle (absolute, NOT relative-to-link1)
  - theta3_rel = theta3_abs - theta2  <- this is what actually gets sent
    to the elbow servo once you have calibration offsets from hardware.

This script only works in math angles (theta2, theta3_abs). It does NOT apply
servo calibration offsets/signs — that conversion happens in a separate step
once you've run the calibration procedure on the real hardware.

Run:
    python3 ik_fk_validator.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# ARM PARAMETERS  -- TODO: replace with measured values once CAD/hardware exist
# ----------------------------------------------------------------------------
L1 = 7.5    # cm, base joint to elbow joint (link 1 length)      -- TODO measure
L2 = 7.5    # cm, elbow joint to gripper tip (link 2 length)     -- TODO measure
H0 = 5.0    # cm, height of shoulder/base joint above the ground -- TODO measure

# Joint travel limits (degrees) -- TODO tighten to actual servo/mechanical limits
THETA1_LIMITS = (-90, 90)
THETA2_LIMITS = (-10, 120)     # relative to horizontal; small negative allowed to reach low targets
THETA3_LIMITS = (-120, 120)    # absolute, same reference as theta2


# ----------------------------------------------------------------------------
# FORWARD KINEMATICS
# ----------------------------------------------------------------------------
def forward_kinematics(theta1_deg, theta2_deg, theta3_abs_deg):
    """Joint angles (degrees, math convention) -> gripper (x, y, z) in cm."""
    t1 = np.radians(theta1_deg)
    t2 = np.radians(theta2_deg)
    t3 = np.radians(theta3_abs_deg)

    r = L1 * np.cos(t2) + L2 * np.cos(t3)
    z = H0 + L1 * np.sin(t2) + L2 * np.sin(t3)
    x = r * np.cos(t1)
    y = r * np.sin(t1)
    return x, y, z


def elbow_joint_position(theta1_deg, theta2_deg):
    """Position of the elbow joint (end of link 1) -- for plotting the arm."""
    t1 = np.radians(theta1_deg)
    t2 = np.radians(theta2_deg)
    r = L1 * np.cos(t2)
    z = H0 + L1 * np.sin(t2)
    return r * np.cos(t1), r * np.sin(t1), z


# ----------------------------------------------------------------------------
# INVERSE KINEMATICS
# ----------------------------------------------------------------------------
class Unreachable(Exception):
    pass


def inverse_kinematics(x, y, z, elbow="up"):
    """
    Target (x, y, z) in cm -> (theta1, theta2, theta3_abs) in degrees.
    elbow: "up" or "down" -- picks between the two valid solutions.
    Raises Unreachable if the target is outside the workspace.
    """
    theta1 = np.degrees(np.arctan2(y, x))

    r = np.hypot(x, y)
    zp = z - H0

    dist_sq = r**2 + zp**2
    dist = np.sqrt(dist_sq)

    if dist > (L1 + L2) or dist < abs(L1 - L2):
        raise Unreachable(
            f"target dist={dist:.2f}cm outside workspace "
            f"[{abs(L1 - L2):.2f}, {L1 + L2:.2f}]cm"
        )

    # law of cosines: angle at the elbow between the two links
    cos_phi = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)  # guard against float rounding at workspace edge
    phi = np.arccos(cos_phi)  # relative elbow bend, 0..pi

    if elbow == "down":
        phi = -phi

    # angle of the line from shoulder to target
    gamma = np.arctan2(zp, r)
    # angle between that line and link 1
    alpha = np.arctan2(L2 * np.sin(phi), L1 + L2 * np.cos(phi))

    theta2 = np.degrees(gamma - alpha)
    theta3_abs = theta2 + np.degrees(phi)

    return theta1, theta2, theta3_abs


def within_limits(theta1, theta2, theta3_abs):
    checks = [
        THETA1_LIMITS[0] <= theta1 <= THETA1_LIMITS[1],
        THETA2_LIMITS[0] <= theta2 <= THETA2_LIMITS[1],
        THETA3_LIMITS[0] <= theta3_abs <= THETA3_LIMITS[1],
    ]
    return all(checks)


# ----------------------------------------------------------------------------
# VALIDATION: round-trip test (IK -> FK should return the original target)
# ----------------------------------------------------------------------------
def validate_target(x, y, z, elbow="up", tol=1e-3, verbose=True):
    try:
        theta1, theta2, theta3_abs = inverse_kinematics(x, y, z, elbow=elbow)
    except Unreachable as e:
        if verbose:
            print(f"  UNREACHABLE  target=({x:.1f},{y:.1f},{z:.1f})  {e}")
        return None

    x2, y2, z2 = forward_kinematics(theta1, theta2, theta3_abs)
    err = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
    limits_ok = within_limits(theta1, theta2, theta3_abs)

    if verbose:
        status = "OK" if err < tol else "FK/IK MISMATCH"
        limit_flag = "" if limits_ok else "  [OUTSIDE JOINT LIMITS]"
        print(
            f"  target=({x:5.1f},{y:5.1f},{z:5.1f})  "
            f"theta1={theta1:7.2f}  theta2={theta2:7.2f}  theta3_abs={theta3_abs:7.2f}  "
            f"err={err:.5f}cm  [{status}]{limit_flag}"
        )

    return dict(
        theta1=theta1, theta2=theta2, theta3_abs=theta3_abs,
        theta3_rel=theta3_abs - theta2,
        error=err, limits_ok=limits_ok,
    )


# ----------------------------------------------------------------------------
# PLOTTING: visualize the arm reaching a target
# ----------------------------------------------------------------------------
def plot_pose(x, y, z, elbow="up", ax=None, show=True):
    result = validate_target(x, y, z, elbow=elbow, verbose=False)
    if result is None:
        print(f"Cannot plot -- ({x},{y},{z}) is unreachable.")
        return

    theta1, theta2 = result["theta1"], result["theta2"]
    ex, ey, ez = elbow_joint_position(theta1, theta2)
    gx, gy, gz = forward_kinematics(theta1, theta2, result["theta3_abs"])

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # base -> elbow -> gripper
    ax.plot([0, ex, gx], [0, ey, gy], [0, ez, gz], "-o", linewidth=3, markersize=6)
    ax.scatter([x], [y], [z], c="red", marker="x", s=80, label="target")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.legend()
    if show:
        plt.show()
    return ax


def plot_workspace_targets(targets):
    """Plot the arm reaching each target in `targets` on one 3D figure."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for (x, y, z) in targets:
        result = validate_target(x, y, z, verbose=False)
        if result is None:
            continue
        theta1, theta2 = result["theta1"], result["theta2"]
        ex, ey, ez = elbow_joint_position(theta1, theta2)
        gx, gy, gz = forward_kinematics(theta1, theta2, result["theta3_abs"])
        ax.plot([0, ex, gx], [0, ey, gy], [0, ez, gz], "-o", alpha=0.6)
        ax.scatter([x], [y], [z], c="red", marker="x", s=40)

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title("Arm reaching test targets")
    plt.savefig("/mnt/user-data/outputs/ik_fk_validation.png", dpi=150)
    print("\nSaved plot -> ik_fk_validation.png")


# ----------------------------------------------------------------------------
# MAIN: run a battery of test targets through the round-trip check
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Arm params: L1={L1}cm  L2={L2}cm  H0={H0}cm  "
          f"max reach={L1+L2:.1f}cm  min reach={abs(L1-L2):.1f}cm\n")

    test_targets = [
        (10.0, 0.0, 5.0),    # straight ahead, mid-height
        (0.0, 10.0, 5.0),    # 90deg to the side
        (7.0, 7.0, 8.0),     # diagonal, higher up
        (14.9, 0.0, 5.0),    # near full extension (edge of workspace)
        (2.0, 0.0, 5.0),     # close in (near min reach -- may be unreachable)
        (10.0, 0.0, 0.0),    # ground level
        (20.0, 0.0, 5.0),    # deliberately outside workspace
    ]

    print("Round-trip validation (elbow-up):")
    for (x, y, z) in test_targets:
        validate_target(x, y, z, elbow="up")

    print("\nRound-trip validation (elbow-down), same targets:")
    for (x, y, z) in test_targets:
        validate_target(x, y, z, elbow="down")

    reachable_targets = [t for t in test_targets
                          if validate_target(*t, verbose=False) is not None]
    plot_workspace_targets(reachable_targets)

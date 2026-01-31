import time
from collections import defaultdict

class GroupManager:
    """
    Handles manual group registration and ownership management.
    This version is designed for manual entry (no images).
    """

    def __init__(self):
        # Core structures
        self.groups = {}                   # group_id -> set(person_ids)
        self.person_to_group = {}           # person_id -> group_id
        self.group_objects = defaultdict(set)  # group_id -> set(object_ids)
        self.object_to_group = {}           # object_id -> group_id
        self.primary_owner = {}             # object_id -> person_id (owner)

        # Temporary data for registration
        self.registration_mode = False
        self.registration_data = {'people': set(), 'objects': {}}
        self.next_group_id = 1

    # ----------------------------------------------------------------------
    # 1️⃣ REGISTRATION PHASE
    # ----------------------------------------------------------------------

    def start_registration(self):
        """Start a new manual registration session."""
        self.registration_mode = True
        self.registration_data = {'people': set(), 'objects': {}}
        print("🔵 Manual registration started - add people and their belongings.")

    def register_person(self, person_id):
        """Add a person by ID."""
        if not self.registration_mode:
            print("⚠️ Not in registration mode.")
            return
        self.registration_data['people'].add(person_id)
        print(f"✅ Registered Person #{person_id}")

    def register_object(self, person_id, object_id, object_name):
        """Link an object to a person."""
        if not self.registration_mode:
            print("⚠️ Not in registration mode.")
            return

        self.registration_data['objects'][object_id] = {
            'owner': person_id,
            'name': object_name.strip(),
            'timestamp': time.time()
        }
        print(f"📦 Registered '{object_name}' (ID:{object_id}) for Person #{person_id}")

    def complete_registration(self):
        """Finalize and create a new group."""
        if not self.registration_mode:
            print("⚠️ No registration session active.")
            return None

        if len(self.registration_data['people']) == 0:
            print("⚠️ No people added to this group.")
            return None

        group_id = self.next_group_id
        self.next_group_id += 1

        # Store group members
        self.groups[group_id] = set(self.registration_data['people'])
        for pid in self.registration_data['people']:
            self.person_to_group[pid] = group_id

        # Store object ownership
        for obj_id, data in self.registration_data['objects'].items():
            self.group_objects[group_id].add(obj_id)
            self.object_to_group[obj_id] = group_id
            self.primary_owner[obj_id] = data['owner']

        self.registration_mode = False

        print(f"\n🎯 Group #{group_id} created successfully!")
        print(f"   Members: {len(self.groups[group_id])} | Objects: {len(self.group_objects[group_id])}")
        return group_id

    # ----------------------------------------------------------------------
    # 2️⃣ CHECK OWNERSHIP & AUTHORIZATION
    # ----------------------------------------------------------------------

    def is_authorized_transfer(self, object_id, from_person_id, to_person_id):
        """
        Check if an object transfer between two people is authorized.
        Returns (is_authorized, reason, alert_type)
        """
        # Check if object exists
        if object_id not in self.object_to_group:
            return False, "Unregistered object", "unregistered_object"

        obj_group = self.object_to_group[object_id]
        from_group = self.person_to_group.get(from_person_id)
        to_group = self.person_to_group.get(to_person_id)

        # Same group = authorized
        if from_group == obj_group and to_group == obj_group:
            return True, "Authorized - Same group", "authorized"
        elif to_group == obj_group:
            return True, "Authorized - Group member", "authorized"
        elif from_group == obj_group:
            return False, "Theft - Object leaving the group", "theft"
        else:
            return False, "Unauthorized person", "theft"

    # ----------------------------------------------------------------------
    # 3️⃣ HELPER FUNCTIONS
    # ----------------------------------------------------------------------

    def get_group_members(self, group_id):
        """Return members in a specific group."""
        return self.groups.get(group_id, set())

    def get_person_group(self, person_id):
        """Return the group ID a person belongs to."""
        return self.person_to_group.get(person_id)

    def get_object_group(self, object_id):
        """Return the group ID an object belongs to."""
        return self.object_to_group.get(object_id)

    def get_primary_owner(self, object_id):
        """Return the original owner of an object."""
        return self.primary_owner.get(object_id)

    def is_group_object(self, object_id):
        """Check if an object is part of any group."""
        return object_id in self.object_to_group

    def get_group_summary(self):
        """Get a formatted summary of all registered groups."""
        if len(self.groups) == 0:
            return "⚠️ No groups registered yet."

        summary = f"📊 Registered Groups: {len(self.groups)}\n"
        for gid, members in self.groups.items():
            summary += f"\nGroup #{gid}:\n"
            summary += f"  Members: {sorted(list(members))}\n"
            summary += f"  Objects: {len(self.group_objects[gid])} items\n"
        return summary

import math


class Animator:
    STEP_SIZE = 20

    def __init__(self, canvas):
        self._canvas = canvas
        pentagon = [
            677, 283,
            777, 283,
            827, 383,
            727, 483,
            627, 383,
            677, 283
        ]
        self._pentagon_id = self._canvas.create_polygon(pentagon, outline='black', fill='yellow', width=2)
        self.traced = []

    @staticmethod
    def _rotate_point_around_point(point, center, angle):
        angle_rad = math.radians(angle)
        ox, oy = center
        px, py = point
        qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
        qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
        return qx, qy

    def move_left(self):
        coords = self._canvas.coords(self._pentagon_id)
        if self._boundary_check(coords):
            new_coords = [coords[i] - (Animator.STEP_SIZE if not i % 2 else 0) for i in range(len(coords))]
            if self._boundary_check(new_coords):
                self.traced.append(self._canvas.create_polygon(coords, outline='black', fill='white', width=2))
                self._canvas.coords(self._pentagon_id, *new_coords)
                self._canvas.tag_raise(self._pentagon_id)
                self._canvas.after(50, lambda: self.move_left())

    def _boundary_check(self, coords) -> bool:
        return (min(coords[::2]) >= 0 and max(coords[::2]) <= self._canvas.winfo_width()
                and min(coords[1::2]) >= 0 and max(coords[1::2]) <= self._canvas.winfo_height())

    def empty_traces(self):
        for self._pentagon_id in self.traced:
            self._canvas.delete(self._pentagon_id)
        self.traced.clear()

    def reset(self):
        canvas_width = self._canvas.winfo_width()
        canvas_height = self._canvas.winfo_height()
        center_x, center_y = canvas_width / 2, canvas_height / 2

        coords = self._canvas.coords(self._pentagon_id)
        pentagon_center_x = sum(coords[::2]) / (len(coords) / 2)
        pentagon_center_y = sum(coords[1::2]) / (len(coords) / 2)

        translation_x = center_x - pentagon_center_x
        translation_y = center_y - pentagon_center_y

        new_coords = [coords[i] + (translation_x if i % 2 == 0 else translation_y) for i in range(len(coords))]
        self._canvas.coords(self._pentagon_id, *new_coords)
        self.empty_traces()

    def rotate(self, spin_angle, orbit_angle, number_of_spins):
        if number_of_spins >= 360:
            return

        coords = self._canvas.coords(self._pentagon_id)

        centroid_x, centroid_y = Animator._get_centroid(coords)
        center_x, center_y = self.get_center()

        spun_coords = []
        for i in range(0, len(coords), 2):
            x, y = Animator._rotate_point_around_point((coords[i], coords[i + 1]), (centroid_x, centroid_y), spin_angle)
            spun_coords.extend([x, y])

        new_centroid_x, new_centroid_y = Animator._get_centroid(spun_coords)
        orbit_x, orbit_y = Animator._rotate_point_around_point(
            (new_centroid_x, new_centroid_y), (center_x, center_y), orbit_angle)

        translation_x = orbit_x - new_centroid_x
        translation_y = orbit_y - new_centroid_y

        final_coords = [spun_coords[i] + (translation_x if i % 2 == 0 else translation_y) for i in
                        range(len(spun_coords))]

        self.traced.append(self._canvas.create_polygon(coords, outline='black', fill='white', width=2))
        self._canvas.coords(self._pentagon_id, *final_coords)
        self._canvas.tag_raise(self._pentagon_id)
        self._canvas.after(25, lambda: self.rotate(spin_angle, orbit_angle, number_of_spins + 1))

    def move_up(self):
        coords = self._canvas.coords(self._pentagon_id)
        if self._boundary_check(coords):
            new_coords = [coords[i] - (Animator.STEP_SIZE if i % 2 else 0) for i in range(len(coords))]
            if self._boundary_check(new_coords):
                self.traced.append(self._canvas.create_polygon(coords, outline='black', fill='white', width=2))
                self._canvas.coords(self._pentagon_id, *new_coords)
                self._canvas.tag_raise(self._pentagon_id)
                self._canvas.after(50, lambda: self.move_up())

    def move_down(self):
        coords = self._canvas.coords(self._pentagon_id)
        if self._boundary_check(coords):
            new_coords = [coords[i] + (Animator.STEP_SIZE if i % 2 else 0) for i in range(len(coords))]
            if self._boundary_check(new_coords):
                self.traced.append(self._canvas.create_polygon(coords, outline='black', fill='white', width=2))
                self._canvas.coords(self._pentagon_id, *new_coords)
                self._canvas.tag_raise(self._pentagon_id)
                self._canvas.after(50, lambda: self.move_down())

    def move_right(self):
        coords = self._canvas.coords(self._pentagon_id)
        if self._boundary_check(coords):
            new_coords = [coords[i] + (Animator.STEP_SIZE if not i % 2 else 0) for i in range(len(coords))]
            if self._boundary_check(new_coords):
                self.traced.append(self._canvas.create_polygon(coords, outline='black', fill='white', width=2))
                self._canvas.coords(self._pentagon_id, *new_coords)
                self._canvas.tag_raise(self._pentagon_id)
                self._canvas.after(50, lambda: self.move_right())

    def zoom(self, scale_factor):
        coords = self._canvas.coords(self._pentagon_id)
        center_x = sum(coords[::2]) / (len(coords) / 2)
        center_y = sum(coords[1::2]) / (len(coords) / 2)
        new_coords = []
        for i in range(len(coords)):
            if i % 2 == 0:
                new_coord = center_x + (coords[i] - center_x) * scale_factor
            else:
                new_coord = center_y + (coords[i] - center_y) * scale_factor
            new_coords.append(new_coord)
        if self._boundary_check(new_coords):
            self._canvas.coords(self._pentagon_id, *new_coords)

    @staticmethod
    def _get_centroid(coords):
        centroid_x = sum(coords[::2]) / (len(coords) / 2)
        centroid_y = sum(coords[1::2]) / (len(coords) / 2)
        return centroid_x, centroid_y

    def get_center(self):
        center_x = self._canvas.winfo_width() / 2
        center_y = self._canvas.winfo_height() / 2
        return center_x, center_y
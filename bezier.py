class BezierCurve:
    def __init__(self, points: list):
        if len(points) < 2:
            raise ValueError('Must be at least 2 points')
        self.points = points

    def curve(self, t):
        points = []
        for _1 in range(len(self.points) - 1):
            start, end = self.points[_1], self.points[_1 + 1]
            point = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
            )
            points.append(point)

        if len(points) == 1:
            return points[0]

        return BezierCurve(points).curve(t)

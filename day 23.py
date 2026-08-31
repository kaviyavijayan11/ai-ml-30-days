import sqlite3
import pandas as pd
conn = sqlite3.connect(':memory:')
df = pd.read_csv("/content/Student_performance_data _.csv")
df.to_sql('students', conn, index=False, if_exists='replace')
print("Database ready!")
print(f"Total rows: {len(df)}")
def run_query(query):
    return pd.read_sql_query(query, conn)

query1 = """
SELECT StudentID, Age, GPA, GradeClass
FROM students
WHERE GPA >= 3.5
ORDER BY GPA DESC
LIMIT 5
"""
print("Top 5 High GPA Students:")
print(run_query(query1))

query2 = """
SELECT StudentID, StudyTimeWeekly, Absences, GPA
FROM students
WHERE Absences > 20 AND GPA < 1.5
"""
print("\nAt-Risk Students:")
print(run_query(query2))

query1 = """
SELECT
    COUNT(*) as total_students,
    AVG(GPA) as avg_gpa,
    MAX(GPA) as max_gpa,
    MIN(GPA) as min_gpa
FROM students
"""
print("Overall Statistics:")
print(run_query(query1))

query2 = """
SELECT
    Gender,
    COUNT(*) as student_count,
    AVG(GPA) as avg_gpa,
    AVG(Absences) as avg_absences
FROM students
GROUP BY Gender
"""
print("\nGender-wise Summary:")
print(run_query(query2))

query3 = """
SELECT
    GradeClass,
    COUNT(*) as count,
    AVG(StudyTimeWeekly) as avg_study_time
FROM students
GROUP BY GradeClass
HAVING COUNT(*) > 100
ORDER BY GradeClass
"""
print("\nGrades with more than 100 students:")
print(run_query(query3))

query1 = """
SELECT
    StudentID,
    GPA,
    CASE
        WHEN GPA >= 3.5 THEN 'Excellent'
        WHEN GPA >= 2.5 THEN 'Good'
        WHEN GPA >= 1.5 THEN 'Average'
        ELSE 'Poor'
    END as Performance
FROM students
LIMIT 10
"""
print("Performance Categories:")
print(run_query(query1))

query2 = """
SELECT
    CASE
        WHEN Absences <= 5 THEN 'Low'
        WHEN Absences <= 15 THEN 'Medium'
        ELSE 'High'
    END as AbsenceLevel,
    COUNT(*) as count,
    AVG(GPA) as avg_gpa
FROM students
GROUP BY AbsenceLevel
"""
print("\nAbsence Level Analysis:")
print(run_query(query2))

import pandas as pd

tutoring_data = pd.DataFrame({
    'TutoringCenter': [1, 0],
    'CenterName': ['Bright Minds Tutoring', 'No Tutoring'],
    'MonthlyFee': [2000, 0]
})
tutoring_data.to_sql('tutoring_centers', conn, index=False, if_exists='replace')

print("Tutoring Centers Table:")
print(tutoring_data)

query1 = """
SELECT
    s.StudentID,
    s.GPA,
    t.CenterName,
    t.MonthlyFee
FROM students s
INNER JOIN tutoring_centers t
ON s.Tutoring = t.TutoringCenter
LIMIT 5
"""
print("\nINNER JOIN Result:")
print(run_query(query1))

query2 = """
SELECT
    t.CenterName,
    COUNT(s.StudentID) as student_count,
    AVG(s.GPA) as avg_gpa
FROM students s
INNER JOIN tutoring_centers t
ON s.Tutoring = t.TutoringCenter
GROUP BY t.CenterName
"""
print("\nGPA by Tutoring Status:")
print(run_query(query2))

query1 = """
SELECT
    StudentID,
    GPA,
    StudyTimeWeekly
FROM students
WHERE Tutoring = 1
  AND ParentalSupport >= 3
ORDER BY GPA DESC
LIMIT 5
"""

print("Query 1 Results:")
print(run_query(query1))

query2 = """
SELECT
    CASE
        WHEN StudyTimeWeekly > 15 THEN 'High Study'
        WHEN StudyTimeWeekly >= 10 THEN 'Medium Study'
        ELSE 'Low Study'
    END AS category,
    COUNT(*) AS count,
    AVG(GPA) AS avg_gpa,
    AVG(Absences) AS avg_absences
FROM students
GROUP BY category
ORDER BY category
"""

print("\nQuery 2 Results:")
print(run_query(query2))

query3 = """
SELECT
    t.CenterName,
    AVG(s.GPA) AS avg_gpa
FROM students s
INNER JOIN tutoring_centers t
    ON s.Tutoring = t.TutoringCenter
GROUP BY t.CenterName
ORDER BY t.TutoringCenter DESC
"""

print("\nQuery 3 Results:")
print(run_query(query3))

query4_count = """
SELECT COUNT(*) AS at_risk_count
FROM students
WHERE Absences > 15
  AND GPA < 1.5
  AND Tutoring = 0
"""

count_result = run_query(query4_count)

query4_students = """
SELECT StudentID
FROM students
WHERE Absences > 15
  AND GPA < 1.5
  AND Tutoring = 0
LIMIT 5
"""

students_result = run_query(query4_students)

print("\nQuery 4 Results:")
print(
    f"Count of at-risk students without tutoring: "
    f"{count_result.iloc[0]['at_risk_count']}"
)
print(
    f"First 5: "
    f"{students_result['StudentID'].tolist()}"
)
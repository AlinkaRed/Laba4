# Отчет по задаче № 1 **Нахождение минимального элемента в векторе**, вариант # 4
# Студент: Редькина Алина Александровна
# Группа: 3823Б1ПР1
# Преподаватель: Сысоев Александр Владимирович, доцент

## Введение

Поиск минимального элемента в массиве данных является одной из фундаментальных задач в программировании и вычислительной математике. Данная операция широко используется в различных алгоритмах. В контексте параллельного программирования эта задача представляет интерес для исследования эффективности различных подходов к распараллеливанию.

## Постановка задачи

**Цель работы:** Реализовать и сравнить последовательную и параллельную версии алгоритма поиска минимального элемента в векторе целых чисел.

**Формальная постановка:** Для заданного вектора V необходимо найти минимальный элемент.

**Требования к реализации:**
- Создать последовательную (SEQ) версию алгоритма.
- Создать параллельную версию с использованием MPI.
- Обеспечить корректность работы алгоритмов для различных типов входных данных.
- Провести сравнительный анализ производительности.

## Описание алгоритма

### Последовательный алгоритм

1. Инициализировать переменную `min_val` первым элементом вектора.
2. Для каждого последующего элемента вектора:
   - Сравнить текущий элемент с `min_val`.
   - Если текущий элемент меньше `min_val`, обновить `min_val`.
3. Вернуть `min_val`.

**Сложность алгоритма:** O(n), где n - размер вектора.

### Параллельный алгоритм (MPI)

1. **Распределение данных:** Главный процесс (rank 0) распределяет данные между всеми процессами.
2. **Локальный поиск:** Каждый процесс находит минимум в своей части данных.
3. **Глобальная редукция:** Используется операция `MPI_Allreduce` с операцией `MPI_MIN` для нахождения глобального минимума.

## Описание схемы параллельного алгоритма

### Схема распределения данных

- Процесс 0: [v[0], v[1], ..., v[k-1]]
- Процесс 1: [v[k], v[k+2], ..., v[2k-1]]
- ...
- Процесс m-1: [v[(m-1)*k], ..., v[n-1]],
где k = n/m (при равномерном распределении)

### Алгоритм работы:

1. Инициализация MPI
2. Процесс 0: рассылает размер вектора всем процессам.
3. Каждый процесс вычисляет свою часть данных.
4. Локальный поиск минимума в своей части.
5. Глобальная редукция для нахождения общего минимума.
6. Процесс 0: возвращает результат.


## Описание программной реализации

### Структура проекта

Проект организован в виде нескольких модулей:

- **common.hpp** - общие определения типов данных.
- **ops_seq.hpp**, **ops_seq.cpp** - последовательная реализация.
- **ops_mpi.hpp**, **ops_mpi.cpp** - MPI реализация.
- **functional** - функциональное тестирование.
- **perfomance** - тесты производительности.

### Ключевые компоненты:

#### Последовательная версия (RedkinaAMinElemVecSEQ)

'''
bool RedkinaAMinElemVecSEQ::RunImpl() {
  const auto& vec = GetInput();
  if (vec.empty()) {
    GetOutput() = 0;
    return true;
  }

  int min_val = vec[0];
  for (size_t i = 1; i < vec.size(); i++) {
    if (vec[i] < min_val) {
      min_val = vec[i];
    }
  }
  
  GetOutput() = min_val;
  return true;
}
'''

#### MPI версия (RedkinaAMinElemVecMPI)

bool RedkinaAMinElemVecMPI::RunImpl() {
  const auto& vec = GetInput();
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = vec.size();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = n / size;
  int remainder = n % size;
  
  int start_idx, end_idx;
  
  if (rank < remainder) {
    start_idx = rank * (local_size + 1);
    end_idx = start_idx + local_size + 1;
  } else {
    start_idx = remainder * (local_size + 1) + (rank - remainder) * local_size;
    end_idx = start_idx + local_size;
  }

  int local_min = INT_MAX;
  for (int i = start_idx; i < end_idx && i < n; i++) {
    if (vec[i] < local_min) {
      local_min = vec[i];
    }
  }

  int global_min;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

## Результаты экспериментов

### Функциональное тестирование
Были проведены комплексные тесты для проверки корректности работы алгоритмов:

- Положительные числа: {5, 2, 8, 1, 9, 3}
- Отрицательные числа: {-5, -2, -8, -1, -9, -3}
- Смешанные числа: {5, -2, 0, -1, 9, -3}
- Единичный элемент: {42}
- Пустой вектор: {}
- Все одинаковые элементы: {7, 7, 7, 7, 7}
- Дубликаты: {5, 2, 5, 1, 2, 1}
- Большие числа: {1000, 500, 2000, 100, 3000}
- С нулевыми значениями: {10, 0, 20, -5, 0, 15}
- Минимум в начале: {-10, 5, 8, 12, 25}
- Минимум в конце: {15, 8, 12, 5, -3}
- Минимум в середине: {15, 8, -5, 12, 10}
- Большой вектор (1000 элементов)
- Граничные значения: {INT_MAX, INT_MIN, 0, -100, 100}
- Чередующиеся знаки: {10, -10, 20, -20, 30, -30}
- Повторяющиеся минимумы: {1, 3, 1, 5, 1}

Результат: Все тесты корректности пройдены успешно, обе реализации возвращают идентичные результаты для одинаковых входных данных.

### Оценка производительности

Для оценки производительности использовались тесты с большим объемом данных (200 000 000 элементов).
Наблюдения:
- Для задачи поиска минимального элемента в векторе MPI версия демонстрирует производительность, сопоставимую с последовательной версией.
- Накладные расходы на межпроцессное взаимодействие компенсируются параллельной обработкой данных.
- Поиск минимума является операцией с низкой вычислительной сложностью (O(n)), поэтому выигрыш от параллелизации может быть незначительным по сравнению с накладными расходами MPI.

## Выводы из результатов

1. **Корректность**: Обе реализации (SEQ и MPI) демонстрируют идентичное поведение и проходят все тесты на корректность.
2. **Производительность**:
   - Для данной задачи параллельная реализация не дает значительного ускорения.
   - Накладные расходы MPI коммуникации сопоставимы с выигрышем от параллельной обработки.
3. **Масштабируемость**: Алгоритм хорошо масштабируется для больших объемов данных, но эффективность параллелизации ограничена природой задачи.
4. **Практическая применимость**: Параллельная версия может быть полезна в случаях, когда поиск минимума является частью более сложной параллельной задачи.

## Заключение
В ходе работы были успешно реализованы последовательная и параллельная версии алгоритма поиска минимального элемента в векторе. Обе реализации прошли полное тестирование на корректность и показали идентичные результаты.

Основной вывод работы заключается в том, что для задач с низкой вычислительной сложностью, таких как поиск минимума, использование MPI может не давать значительного прироста производительности из-за накладных расходов на межпроцессную коммуникацию.

## Список литературы
1. Лекции Сысоева Александра Владимировича.

## Приложение

Основной код реализации

common.hpp
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace redkina_a_min_elem_vec {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace redkina_a_min_elem_vec


ops_seq.hpp
#include "redkina_a_min_elem_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <climits>

#include "redkina_a_min_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace redkina_a_min_elem_vec {

RedkinaAMinElemVecSEQ::RedkinaAMinElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RedkinaAMinElemVecSEQ::ValidationImpl() {
  return (GetOutput() == 0);
}

bool RedkinaAMinElemVecSEQ::PreProcessingImpl() {
  return true;
}

bool RedkinaAMinElemVecSEQ::RunImpl() {
  const auto& vec = GetInput();
  if (vec.empty()) {
    GetOutput() = 0;
    return true;
  }

  int min_val = vec[0];
  for (size_t i = 1; i < vec.size(); i++) {
    if (vec[i] < min_val) {
      min_val = vec[i];
    }
  }
  
  GetOutput() = min_val;
  return true;
}

bool RedkinaAMinElemVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_min_elem_vec


ops_mpi.hpp
#include "redkina_a_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>
#include <climits>

#include "redkina_a_min_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace redkina_a_min_elem_vec {

RedkinaAMinElemVecMPI::RedkinaAMinElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RedkinaAMinElemVecMPI::ValidationImpl() {
  return (GetOutput() == 0);
}

bool RedkinaAMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool RedkinaAMinElemVecMPI::RunImpl() {
  const auto& vec = GetInput();
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = vec.size();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  int local_size = n / size;
  int remainder = n % size;
  
  int start_idx, end_idx;
  
  if (rank < remainder) {
    start_idx = rank * (local_size + 1);
    end_idx = start_idx + local_size + 1;
  } else {
    start_idx = remainder * (local_size + 1) + (rank - remainder) * local_size;
    end_idx = start_idx + local_size;
  }

  int local_min = INT_MAX;
  for (int i = start_idx; i < end_idx && i < n; i++) {
    if (vec[i] < local_min) {
      local_min = vec[i];
    }
  }

  if (local_size == 0 && rank >= n) {
    local_min = INT_MAX;
  }

  int global_min;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;

  return true;
}

bool RedkinaAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_min_elem_vec

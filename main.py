import os
from itertools import zip_longest
from math import ceil
from queue import Queue
from threading import Thread

import numpy as np
from gmpy2 import mpz


class NumberFile:
    """
    A class for a file contains only a number.

    Constraint 1: The file should not start with 0.
    Constraint 2: The file should contain only 0-9 or '.'.
    Constraint 3: The file should not contain more than one '.'.
    """

    def __init__(self, filepath: str):
        self.filepath: str = filepath

        self.size: int = os.path.getsize(self.filepath)

        self.decimal_pt_idx: int = -1
        self.num_decimal_places: int = -1

    def read_chunks(self, chunk_size: int) -> str:
        """
        A generator function reads the file sequentially and yields data chunks
        """
        with open(self.filepath, 'r') as f:
            while data_chunk := f.read(chunk_size):
                yield data_chunk

    def read_chunks_with_buffer(self, chunk_size: int) -> str:
        def prefetch_thread():
            for data_chunk_ in self.read_chunks(chunk_size):
                buffer_queue.put(data_chunk_)
            buffer_queue.put(None)

        # Create a buffer queue
        buffer_queue = Queue(maxsize=1)

        # Start data fetching in a separate thread
        Thread(target=prefetch_thread, daemon=True).start()

        # Yield data chunks from the queue
        while data_chunk := buffer_queue.get():
            yield data_chunk

    def slice(self, start: int, end: int) -> str:
        if start < 0:
            start += self.size
        if end < 0:
            end += self.size

        # Clip the value to be within [0, self.size]
        start = np.clip(start, a_min=0, a_max=self.size)
        end = np.clip(end, a_min=0, a_max=self.size)

        offset = start
        count = max(0, end - start)

        with open(self.filepath, 'r') as f:
            f.seek(offset)
            return f.read(count)

    def validate(self, chunk_size) -> None:
        """
        Check if the file conforms to the constraints, and get 'num_decimal_places' by the way.
        """
        valid_char_set = set(map(str, range(10)))
        valid_char_set.add('.')

        for i, data_chunk in enumerate(self.read_chunks_with_buffer(chunk_size)):
            # Constraint 1
            if i == 0 and data_chunk[0] == '0':
                raise AssertionError("The file should not start with 0.")

            # Constraint 2
            data_chunk_char_set = set(data_chunk)
            if not valid_char_set >= data_chunk_char_set:
                raise AssertionError("The file should contain only 0-9 or '.'.")

            if '.' in data_chunk_char_set:
                # Constraint 3
                self.decimal_pt_idx = chunk_size * i + data_chunk.index('.')
                if self.num_decimal_places != -1 or '.' in data_chunk[self.decimal_pt_idx + 1:]:
                    raise AssertionError("The file should not contain more than one '.'.")

                self.num_decimal_places = self.size - (self.decimal_pt_idx + 1)


class FileScaleMultiplier:
    def __init__(self,
                 filepath_1: str,
                 filepath_2: str,
                 chunk_size: int = 10 ** 9,  # 1 GB
                 ):
        self.files = [  # [0]: multiplicand; [1]: multiplier
            NumberFile(filepath_1),
            NumberFile(filepath_2),
        ]
        self.chunk_size = chunk_size

        assert self.chunk_size >= 1

    def multiply(self):
        def prefetch_thread(file_idx, start_chunk, end_chunk):
            """
            A thread function fetches data chunks and puts them into a queue.
            The decimal point is removed automatically.

            :param file_idx: The index of the file to be read.
            :param start_chunk: An index within [0, num_chunks), where 0 represents the first chunk when counting from
            the end of the file.
            :param end_chunk: Same logic as 'start_chunk', and the 'end_chunk' is also included in the output.
            """
            # Create aliases to improve readability
            file = self.files[file_idx]
            queue = buffer_queues[file_idx]

            # Ensure parameter values are valid
            assert 0 <= start_chunk < num_chunks_list[file_idx]
            assert 0 <= end_chunk < num_chunks_list[file_idx]

            step = int(start_chunk <= end_chunk) or -1
            for chunk_idx in range(start_chunk, end_chunk + step, step):
                # Calculate the start and end indices
                start_idx = file.size - self.chunk_size * (chunk_idx + 1)
                end_idx = start_idx + self.chunk_size

                # Shift the indices to skip the decimal point
                contain_decimal_pt = False
                if end_idx <= file.decimal_pt_idx:
                    # The decimal point is located after this chunk
                    start_idx -= 1
                    end_idx -= 1
                elif start_idx <= file.decimal_pt_idx:
                    # The decimal point is located within this chunk
                    start_idx -= 1
                    contain_decimal_pt = True

                # Ensure start_idx >= 0
                start_idx = max(0, start_idx)

                data_chunk = file.slice(start_idx, end_idx)
                queue.put(data_chunk.replace('.', '') if contain_decimal_pt else data_chunk)

            queue.put(None)

        # Create a temporary folder for storing intermediate files
        os.makedirs("./tmp", exist_ok=True)

        # Create buffer queues
        buffer_queues = [  # [0]: multiplicand; [1]: multiplier
            Queue(maxsize=1),
            Queue(maxsize=1),
        ]

        # Calculate the number of chunks for each input file (excluding the decimal point)
        num_chunks_list = [ceil((file.size - (file.decimal_pt_idx != -1)) / self.chunk_size) for file in self.files]

        # Record the index of the target multiplicand chunk for the multiplier chunks
        target_chunk_idx_arr = -np.arange(num_chunks_list[1])

        i, carry = 0, 0
        while (target_chunk_idx_arr < num_chunks_list[0]).any():
            valid_multiplier_mask = (0 <= target_chunk_idx_arr) & (target_chunk_idx_arr < num_chunks_list[0])
            valid_multiplier_idx_arr = np.where(valid_multiplier_mask)[
                0]  # The indexes are always contiguous and in ascending order
            multiplier_start_chunk = valid_multiplier_idx_arr[0]
            multiplier_end_chunk = valid_multiplier_idx_arr[-1]

            multiplicand_start_chunk = target_chunk_idx_arr[valid_multiplier_mask][0]
            multiplicand_end_chunk = target_chunk_idx_arr[valid_multiplier_mask][-1]

            # Start threads to concurrently fetch chunk data into the queue
            Thread(target=prefetch_thread, args=(1, multiplier_start_chunk, multiplier_end_chunk), daemon=True).start()
            Thread(target=prefetch_thread, args=(0, multiplicand_start_chunk, multiplicand_end_chunk),
                   daemon=True).start()

            chunk_product_sum = carry
            for multiplicand_chunk_str, multiplier_chunk_str in zip_longest(
                    *[iter(buffer_queue.get, None) for buffer_queue in buffer_queues]):
                chunk_product_sum += mpz(multiplicand_chunk_str) * mpz(multiplier_chunk_str)

            quotient, remainder = divmod(chunk_product_sum, mpz(10) ** self.chunk_size)
            carry = quotient

            with open(f"./tmp/{i:08d}.chunk", 'w') as f:
                f.write(str(remainder).zfill(self.chunk_size))
                print("Written:", f.name)

            target_chunk_idx_arr += 1
            i += 1

        if carry:
            with open(f"./tmp/{i:08d}.chunk", 'w') as f:
                f.write(str(carry))
                print("Written:", f.name)

    def merge_results(self):
        result_filepath = "./result.txt"

        # Get intermediate file list
        intermediate_filepaths = sorted([f"./tmp/{filename}" for filename in os.listdir("./tmp")
                                         if filename.endswith('.chunk')], reverse=True)

        # Calculate the total number of digits written (including leading zeros)
        total_num_digits = sum(os.path.getsize(filepath) for filepath in intermediate_filepaths)

        # Calculate the number of decimal places in the result
        result_num_decimal_places = sum(file.num_decimal_places for file in self.files if file.num_decimal_places != -1)

        # Fix an edge case, e.g., 0.1 * 0.1
        if result_num_decimal_places > total_num_digits:
            with open(f"./tmp/{len(intermediate_filepaths):08d}.chunk", 'w') as f:
                f.write('0' * (result_num_decimal_places - total_num_digits))
                print("Written:", f.name)

            return self.merge_results()

        # Calculate the index of the decimal point in the result
        result_decimal_pt_idx = total_num_digits - result_num_decimal_places if result_num_decimal_places else -1

        # Create the result file
        with open(result_filepath, 'w') as f:
            print("Created:", f.name)

        total_digits_written = 0
        for i, filepath in enumerate(intermediate_filepaths):
            # Read the intermediate file
            with open(filepath, 'r') as f:
                data = f.read()

            # Insert the decimal point
            if total_digits_written <= result_decimal_pt_idx < total_digits_written + len(data):
                insert_idx = result_decimal_pt_idx - total_digits_written
                data = data[:insert_idx] + '.' + data[insert_idx:]

            if i == 0:
                # Remove leading zeros
                original_len = len(data)
                data = data.lstrip('0')
                num_digits_removed = original_len - len(data)
                result_decimal_pt_idx -= num_digits_removed

            # Append the data chunk to the result file
            with open(result_filepath, 'a') as f:
                f.write(data)
            total_digits_written += len(data)

            # Remove the intermediate file
            try:
                os.remove(filepath)
            except Exception as e:
                print(e)
            else:
                print("Removed:", filepath)

    def main(self):
        # Validate input files and get 'num_decimal_places'
        for file in self.files:
            file.validate(chunk_size=self.chunk_size)

        # Ensure the multiplicand is longer
        self.files.sort(key=lambda file_: file_.size - (file_.decimal_pt_idx != -1), reverse=True)

        # Multiply two files chunk by chunk
        self.multiply()

        # Merge the intermediate result files into one file
        self.merge_results()

        # Remove the temporary folder
        try:
            os.rmdir("./tmp")
        except Exception as e:
            print(e)
        else:
            print("Removed:", "./tmp")


if __name__ == '__main__':
    FileScaleMultiplier(
        "./input_file_1.txt",
        "./input_file_2.txt",
        chunk_size=10,  # debug
    ).main()

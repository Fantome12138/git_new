'''
03 数组中的重复数字 *** / Array 

在一个长度为n的数组里的所有数字都在0到n-1的范围内。数组中某些数字是重复的，
但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 
例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
'''
# class Solution:
#     def Find_repeat1(self, nums):
#         Hashset = set()
#         for num in nums:
#             if num in Hashset:return num
#             Hashset.add(num)
#         return Hashset

#     def Find_repeat2(self, nums):
#         nums.sort()
#         for i in range(len(nums)):
#             if nums[i] == nums[i-1]:
#                 return nums[i]
#         return -1
    
#     def Find_repeat3(self, nums):
#         i = 0
#         while i < len(nums):
#             if nums[i] == i:
#                 i += 1
#                 continue
#             if nums[nums[i]] == nums[i]:return nums[i]
#             nums[nums[i]], nums[i] = nums[nums[i]], nums[i]
#         return -1

# nums = [2,3,1,0,2,5,3]
# a = Solution()
# print(a.Find_repeat1(nums))
# print(a.Find_repeat2(nums))
# print(a.Find_repeat3(nums))

'''
04 二维数组中的查找 *** / Array 

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。
给定 target = 20，返回 false。
'''
# class Solution:
#     def Find_in2Dmatrix1(self, matrix, nums):
#         if matrix == 0 or len(matrix) == 0:return False
#         i, j = len(matrix)-1, 0
#         while i > 0 and j < len(matrix[0]):
#             if matrix[i][j] > nums: i -= 1
#             elif matrix[i][j] < nums: j += 1
#             else: return True
#         return False


#     def binary_search(self, matrix, target, start, vertical):
#         lo = start
#         hi = len(matrix) - 1 if vertical else len(matrix[0]) - 1 # 垂直搜索：hi = 行数 - 1

#         while lo <= hi:
#             mid = (lo + hi) // 2
#             if vertical:
#                 if matrix[mid][start] < target:
#                     lo = mid + 1
#                 elif matrix[mid][start] > target:
#                     hi = mid - 1
#                 else: return True
#             else:
#                 if matrix[start][mid] < target:
#                     lo = mid + 1
#                 elif matrix[start][mid] > target:
#                     hi = mid - 1
#                 else: return True
#         return False
    
#     def Find_in2Dmatrix2(self, matrix, target):
#         if not matrix: return False   # 边界条件

#         for i in range(min(len(matrix), len(matrix[0]))):
#             vertical_found = self.binary_search(matrix, target, i, True) # 垂直方向是否找到
#             horizontal_found = self.binary_search(matrix, target, i, False) # 水平是否找到
#             if vertical_found or horizontal_found:  # 任一方向找到即可
#                 return True

#         return False
   
# matrix = [
#   [1,   4,  7, 11, 15],
#   [2,   5,  8, 12, 19],
#   [3,   6,  9, 16, 22],
#   [10, 13, 14, 17, 24],
#   [18, 21, 23, 26, 30]
# ]
# a = Solution()
# print(a.Find_in2Dmatrix1(matrix,9))
# print(a.Find_in2Dmatrix2(matrix,9))

'''
05 替换空格 * / String 

请实现一个函数，将一个字符串中的空格替换成“%20”。
例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
'''
# class Solution:
#     def Replace_space1(self, lst):
#         tmp = ''
#         if type(lst) != str:
#             return
#         for item in lst:
#             if item == ' ':
#                 tmp += '%20'
#             else:
#                 tmp += item
#         return tmp

#     def Replace_space2(self, lst):
#         res = []
#         for i in lst:
#             if i == ' ':
#                 res.append('%20')
#             else:
#                 res.append(i)
#         return ''.join(res)
    
#     def Replace_space3(self, lst):
#         lst = lst.split(' ')
#         return '%20'.join(lst)
        
#     def Replace_space4(self, lst):
#         #双指针移动+计数
#         lst_len = len(lst)
#         space_count = 0
#         for i in lst:
#             if i == ' ':
#                 space_count += 1
#             lst_len += 2 * space_count
#             new_array = [' '] * lst_len
#             j = 0
#         for i in range(len(lst)):
#             if lst[i] == ' ':
#                 new_array[j] = '%'
#                 new_array[j+1] = '2'
#                 new_array[j+2] = '0'
#                 j += 3
#             else:
#                 new_array[j] = lst[i]
#                 j += 1
#         return ''.join(new_array)


# lst = 'We Are Happy'
# a = Solution()
# print(a.Replace_space1(lst))
# print(a.Replace_space2(lst))
# print(a.Replace_space3(lst))
# print(a.Replace_space4(lst))


'''
06 从头到尾打印链表 * / Linked List

输入一个链表，从尾到头打印链表每个节点的值。
输入：head = [1,3,2]
输出：[2,3,1]
'''
# class LinkedNode:
#     def __init__(self, data, next=None):
#         self.data = data
#         self.next = next

# class Solution:
#     def reverseLinked1(self, head:LinkedNode):
#         return reverseLinked(head.next) + [head.data] if head else []
    
#     def reverseLinked2(self, head:LinkedNode):
#         stack = []
#         while head: #push
#             stack.append(head.data)
#             head = head.next
#         return stack[::-1]

#     def reverseLinked3(self, head:LinkedNode):
#         stack = []
#         while head: # push
#             stack.append(head.data)
#             head = head.next
#         res = []
#         while stack: # pop
#             res.append(stack.pop())
#         return res

'''
07 重建二叉树 ** / Tree

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。
假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例
前序遍历 preorder = [3,9,20,15,7]   {1,2,4,7,3,5,6,8}
中序遍历 inorder = [9,3,15,20,7]    {4,7,2,1,5,3,8,6}
返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7
'''
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# class Solution:
#     def reConstructBinaryTree1(self, preorder, inorder):
#         if not preorder:
#             None
#         node = TreeNode(preorder[0])
#         index = inorder.index(preorder[0])
#         left_pre = preorder[1:index+1]
#         righr_pre = preorder[index+1:]
#         left_in = inorder[:index]
#         right_in = inorder[index+1:]
#         node.left = self.reConstructBinaryTree1(left_pre, left_in)
#         node.right = self.reConstructBinaryTree1(righr_pre, right_in)
#         return node

#     # 返回构造的TreeNode根节点
#     def reConstructBinaryTree2(self, pre, tin):
#         if not pre and not tin:
#             return None

#         root = TreeNode(pre[0])
#         if set(pre) != set(tin):
#             return None
#         i = tin.index(pre[0])
#         root.left = self.reConstructBinaryTree2(pre[1:i+1], tin[:i])
#         root.right = self.reConstructBinaryTree2(pre[i+1:], tin[i+1:])
#         return root

# pre = [3,9,20,15,7]
# tin = [9,3,15,20,7]
# test = Solution()
# newTree = test.reConstructBinaryTree1(pre, tin)
# newTree = test.reConstructBinaryTree2(pre, tin)
# print(newTree)
# print('>>> ',pre[0])
# print('>>> ',tin.index(pre[0]))

'''
09 使用栈实现队列 ** / Stack, Queue

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead，
分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回-1)
示例 1：
输入：["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
'''
# class Solution(object):
#     def __init__(self):
#         self.stack1 = []
#         self.stack2 = []

#     def appendTail(self, node):
#         self.stack1.append(node)
#         print(self.stack1)

#     def deleteHead(self, node):
#         if len(self.stack2) == 0 and len(self.stack1) == 0:
#             return None
#         elif len(self.stack2) == 0:
#             while len(self.stack1) > 0:
#                 self.stack2.append(self.stack1.pop())
#         return self.stack2.pop()

# P = Solution()
# P.appendTail(10)
# P.appendTail(11)
# P.appendTail(12)
# print(P.deleteHead(1),'\n')
# P.appendTail(13)
# print(P.deleteHead(1))

'''
10 蛙蛙跳台阶 * / Dynamic Programming

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
输入：n = 2
输出：2

输入：n = 7
输出：21

输入：n = 0
输出：1
'''
# class Solution:
#     def Fi(self, n):
#         a, b = 1, 1
#         for _ in range(n):
#             a, b = b, a+b
#         return a%1000000007

# S = Solution()
# print(S.Fi(7))

'''
10 斐波那契数列 * / Dynamic Programming,Math

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
'''
# class Solution:
#     def Fibonacci(self, n):
#         a, b = 0, 1
#         for _ in range(n):
#             a, b = b, a+b
#         return a%1000000007
# S = Solution()
# print(S.Fibonacci(70))

'''
11 旋转数组的最小数字 / 常考***

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，
输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。

思路：通过二分法不断缩小范围，由于mid是整除，最后l==mid，并且nums[mid] > nums[r]的。  
'''
# class Solution:
#       def minArray1(self, nums):
#             low, high = 0, len(nums)-1
#             while low < high:
#                   mid = low + (high - low) // 2  # 不写(high+low)//2，防止high、low数值大时溢出
#                   if nums[mid] < nums[high]:
#                         high = mid
#                   if nums[mid] > nums[high]:
#                         low = mid + 1
#                   else: high -= 1
#             return nums[low]


# nums = [3,4,5,1,2]
# a = Solution()
# print(a.minArray1(nums))

'''
12 矩阵中的路径 * /	BackTracking

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。
例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。
[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]
但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
'''

'''主程序寻找起点，辅助函数用于在给定起点和已探测节点的基础上继续DFS探测，同时用一个字典保留已探测的节点避免重复探测。
当探测节点个数等于目标字符串长度时，即可返回；否则回溯至上一节点。'''

# class Solution:
#       def exit(self, board, word):
#             def DFS(i, j, k):
#                   if not 0<=i<len(board) or not 0<=j<len(board[0]) or board[i][j]!=word[k]: return False
#                   if k == len(word) - 1: return True
#                   tmp, board[i][j] = board[i][j], '/'
#                   res = DFS(i+1,j,k+1) or DFS(i-1,j,k+1) or DFS(i,j+1,k+1) or DFS(i,j-1,k+1)  # 回溯函数
#                   board[i][j] = tmp
#                   return res
#             for i in range(len(board)):
#                   for j in range(len(board[0])):
#                         if DFS(i, j, 0): return True
#             return False


# board = [["a","b","c","e"],
#          ["s","f","c","s"],
#          ["a","d","e","e"]]
# word = ['b','c','c','e']             
# a = Solution()
# print(a.exit(board, word))


'''
13 机器人的运动范围 ** / BackTracking

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，
它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。
例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

示例 1：
输入：m = 2, n = 3, k = 1
输出：3
示例 2：
输入：m = 3, n = 1, k = 0
输出：1
'''
# def digitsum(n):
#       ans = 0
#       while n:
#             ans += n % 10
#             n //= 10
#       return ans

# class Solution:
#       '''DFS'''  
#       def movingCount1(self, m, n, k):
#             def DFS(i, j, si, sj):
#                   if not i <= m or not j <= n or k < (si + sj) or (i, j) in visited: return 0
#                   visited.add((i, j))
#                   return 1 + DFS(i+1, j, si+1 if (i+1)%10 else si-8, sj) + DFS(i, j+1, si, sj+1 if (j+1)%10 else sj-8)
#             visited = set()
#             return DFS(0, 0, 0, 0)

#       '''使用队列实现BFS'''
#       def movingCount2(self, m, n, k):
#             queue, visited = [(0, 0, 0, 0)], set()
#             while queue:
#                   i, j, si, sj = queue.pop(0)
#                   if not i <= m or not j <= n or k < (si + sj) or (i, j) in visited: continue
#                   visited.add((i, j))
#                   queue.append((i+1, j, si+1 if (i+1)%10 else si-8, sj))
#                   queue.append((i, j+1, si, sj+1 if (j+1)%10 else sj-8))
#             return len(visited)

# a = Solution()
# print(a.movingCount1(2, 3, 1))
# print(a.movingCount2(2, 3, 1))

'''
14 剪绳子 *	/ Dynamic Programming

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
示例 2:
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
'''
# class Solution:
#       def cuttingRope1(self, m):
#             if m <= 3: return m - 1 
#             a, b = m // 3, m % 3
#             if b == 0: return 3 ** a
#             if b == 1: return 4 * (3 ** (a-1)) 
#             return 2 * (3 ** a)
#     # 贪心算法
#       def cuttingRope2(self, n: int) -> int:
#           if n == 2:
#               return 1
#           if n == 3:
#               return 2
#           if n % 3 == 0:
#               return 3 ** (n // 3) % int(1e9+7)
#           elif n % 3 == 1:
#               return 3 **((n-4) // 3) * 4 % int(1e9+7)
#           elif n % 3 == 2:
#               return 3 ** ((n-2) // 3) * 2 % int(1e9+7)
# a = Solution()
# print(a.cuttingRope1(13))            
# print(a.cuttingRope2(13))   

'''
15 二进制中1的个数 * / Bit Manipulation

请实现一个函数，输入一个整数，输出该数二进制表示中1的个数。
例如，把9表示成二进制是1001，有2位是1。因此，如果输入9，则该函数输出2。
示例 1：
输入：00000000000000000000000000001011
输出：3
'''
# class Solution:
#     def hammingWeight(self, n):
#         res = 0
#         while n:
#             res += n & 1
#             n >>= 1
#         return res

'''
16.数值的整数次方 ** / Math

实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。
示例 1:
输入: 2.00000, 10
输出: 1024.00000
'''
# class Solution:
#     def doublePower(self, x, n):
#         if x == 0: return 0
#         res = 1
#         if n < 0: x, n = 1 / x, -n
#         while n:
#             if n & 1: res *= x
#             x *= x
#             n >>= 1
#         return res

          
# a = Solution()
# print(a.doublePower(2.00,10))

'''
17 打印从1到最大的n位数 * / 

输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
示例 1:
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
'''
# class Solution:
#       def Print1(self, m):
#           res = []
#           for i in range(1, 10 ** m):
#                 res.append(i)
#           return res



#       def Print2(self, n):
#           def dfs(x):
#               if x == n: # 终止条件：已固定完所有位
#                   res.append(''.join(num)) # 拼接 num 并添加至 res 尾部
#                   return
#               for i in range(10): # 遍历 0 - 9
#                   num[x] = str(i) # 固定第 x 位为 i
#                   dfs(x + 1) # 开启固定第 x + 1 位
          
#           num = ['0'] * n # 起始数字定义为 n 个 0 组成的字符列表
#           res = [] # 数字字符串列表
#           dfs(0) # 开启全排列递归
#           return ','.join(res)  # 拼接所有数字字符串，使用逗号隔开，并返回
      
#       '''改进，去除前面的0'''
#       def Print3(self, n: int) -> [int]:
#             def dfs(x):
#                 if x == n:
#                     s = ''.join(num[self.start:])
#                     if s != '0': res.append(int(s))
#                     if n - self.start == self.nine: self.start -= 1
#                     return
#                 for i in range(10):
#                     if i == 9: self.nine += 1
#                     num[x] = str(i)
#                     dfs(x + 1)
#                 self.nine -= 1

#             num, res = ['0'] * n, []
#             self.nine = 0
#             self.start = n - 1
#             dfs(0)
#             return res

# a = Solution()
# # print(a.Print1(3))
# # print(a.Print2(3))
# print(a.Print3(3))

'''
18 删除链表中重复结点 / 常考***  ????

编写函数以删除单链列表中的节点。您将无权访问head列表的，而是将有权访问要直接删除的节点。
这是保证是要删除的节点不是尾节点在列表中。
输入： head = [4,5,1,9]，node = 5
输出： [4,1,9]说明：第二个节点的值为5，链接列表应变为4-> 1-> 9调用函数后。
'''
# class ListNode:
#     def __init__(self, data, next=None):
#         self.data = data
#         self.next = next

# class Solution:
#     def deleteNode1(self, head:ListNode, val):
#           if head.val == val: return head.next
#           pre, cur = head, head.next
#           while cur and cur.val != val:
#                 pre, cur = cur, cur.next
#           if cur: pre.next = cur.next
#           return head
    
#     '''哨兵节点法'''
#     def deleteNode2(self, head:ListNode, val):
#         sentinel = ListNode(0)
#         sentinel.next = head
#         prev, cur = sentinel, head
#         while cur:
#             if cur.val == val:
#                 prev.next = cur.next
#                 return sentinel.next # 仅删除一个节点，因此删完就走人
#             else:
#                 prev = cur # 移动前继节点
#             cur = cur.next # 移动当前节点

'''
19 正则表达式匹配 *** / String ???

请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。
在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。
示例 1:
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
示例 2:
输入:
s = "aa"
p = "a*"
输出: true
解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母以及字符 . 和 *，无连续的 '*'。
'''
# class Solution:
#       # 回溯
#       def isMatch1(self, s, p):
#             if not p: return not s
#             first_match = bool(s and p[0] in {s[0],'.'})
#             if len(p) >= 2 and p[1] == '*':
#                   return self.isMatch1(s, p[2:]) or \
#                   first_match and self.isMatch1(s[1:], p)
#             else:
#                   return first_match and self.isMatch1(s[1:], p[1:])

#       # 动态规划
#       def isMatch2(self, s, p):
#             # 边界条件，考虑 s 或 p 分别为空的情况
#             if not p: return not s
#             if not s and len(p) == 1: return False

#             m, n = len(s) + 1, len(p) + 1
#             dp = [[False for _ in range(n)] for _ in range(m)]
#             # 初始状态
#             dp[0][0] = True
#             dp[0][1] = False

#             for c in range(2, n):
#                   j = c - 1
#                   if p[j] == '*':
#                         dp[0][c] = dp[0][c - 2]
            
#             for r in range(1,m):
#                   i = r - 1
#                   for c in range(1, n):
#                         j = c - 1
#                   if s[i] == p[j] or p[j] == '.':
#                         dp[r][c] = dp[r - 1][c - 1]
#                   elif p[j] == '*':       # ‘*’前面的字符匹配s[i] 或者为'.'
#                         if p[j - 1] == s[i] or p[j - 1] == '.':
#                               dp[r][c] = dp[r - 1][c] or dp[r][c - 2]
#                         else:                       # ‘*’匹配了0次前面的字符
#                               dp[r][c] = dp[r][c - 2] 
#                   else:
#                         dp[r][c] = False
#             return dp[m - 1][n - 1]


# s = 'aa'
# p = 'a*'
# a = Solution()
# print(a.isMatch1(s, p))
# print(a.isMatch2(s, p))

'''
20 表示数值的字符串 * / String ???

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"-1E-16"、"0123"都表示数值，
但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。
'''
# class Solution:
#     # 傻瓜作死方法
#     def isNumber1(self, s):
#         try:
#             float(s)
#         except ValueError:
#             return False
#         return True

#     # 动态规划-状态转移，有限状态自动机
#     def isNumber2(self, s):
#         states = [
#             { ' ': 0, 's': 1, 'd': 2, '.': 4 }, # 0. start with 'blank'
#             { 'd': 2, '.': 4 } ,                # 1. 'sign' before 'e'
#             { 'd': 2, '.': 3, 'e': 5, ' ': 8 }, # 2. 'digit' before 'dot'
#             { 'd': 3, 'e': 5, ' ': 8 },         # 3. 'digit' after 'dot'
#             { 'd': 3 },                         # 4. 'digit' after 'dot' (‘blank’ before 'dot')
#             { 's': 6, 'd': 7 },                 # 5. 'e'
#             { 'd': 7 },                         # 6. 'sign' after 'e'
#             { 'd': 7, ' ': 8 },                 # 7. 'digit' after 'e'
#             { ' ': 8 }                          # 8. end with 'blank'
#         ]
#         p = 0                           # start with state 0
#         for c in s:
#             if '0' <= c <= '9': t = 'd' # digit
#             elif c in "+-": t = 's'     # sign
#             elif c in "eE": t = 'e'     # e or E
#             elif c in ". ": t = c       # dot, blank
#             else: t = '?'               # unknown
#             if t not in states[p]: return False
#             p = states[p][t]
#         return p in (2, 3, 7, 8)


# num1 = '1.2.2.3'
# num2 = '1.2214'
# a = Solution()
# print(a.isNumber1(num1))
# print(a.isNumber1(num2))
# print(a.isNumber2(num1))
# print(a.isNumber2(num2))



'''
21 调整数组顺序使奇数位于偶数前面 ** / 

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
示例：
输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。
'''
# class Solution:

#       # 双指针
#       def switch_even_odd1(self, num):
#             i, j = 0, len(num)-1
#             while i < j:
#                   while i < j and num[i] & 1 == 1: i += 1
#                   while i < j and num[j] & 1 == 0: j -= 1
#                   num[i], num[j] = num[j], num[i]
#             return num

#       # 单指针
#       def switch_even_odd2(self, nums):
#             i= 0 
#             for j in range(len(nums)):
#                   if nums[j] % 2 == 1:
#                         nums[i],nums[j] = nums[j],nums[i]
#                         i+=1
#             return nums


# num = [2,4,1,3]
# a = Solution()
# print(a.switch_even_odd1(num))
# print(a.switch_even_odd2(num))

'''
22 链表中倒数第k个结点 *** / Linked List ??

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。
例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。
示例：
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
'''
# class Solution:

#       # 快慢双指针
#       def PrintKfromEnd(self, head, k):
#             former, latter = head, head
#             for _ in range(k):
#                   former = former.next
#             while former:
#                   former, latter = former.next, latter.next
#             return latter

'''
24 反转链表 * / Linked List

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
示例:
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
'''

# class Solution:

#       # 双指针
#       def reverseList1(self, head):
#             pre, cur = None, head
#             while cur:
#                   tmp = cur.next
#                   cur.next = pre
#                   pre = cur
#                   cur = tmp
#             return pre
      
#       # 递归解法
#       def reverseLinked2(self, head):  
#             # 递归终止条件是当前为空，或者下一个节点为空
#             if(head == None or head.next == None):
#                   return head
#             # 这里的cur就是最后一个节点
#             cur = self.reverseList(head.next)
#             # 这里请配合动画演示理解
#             # 如果链表是 1->2->3->4->5，那么此时的cur就是5
#             # 而head是4，head的下一个是5，下下一个是空
#             # 所以head.next.next 就是5->4
#             head.next.next = head
#             # 防止链表循环，需要将head.next设置为空
#             head.next = None
#             # 每层递归函数都返回cur，也就是最后一个节点
#             return cur







print('!!!!!!!!!!')





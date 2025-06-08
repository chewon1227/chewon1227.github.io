---
layout: default
title: Paper
permalink: /Paper/
parent: Category
---
<ul>
  {% assign cat = site.categories.Paper %}
  {% for post in cat %}
    <li><a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date: "%Y-%m-%d" }})</li>
  {% endfor %}
</ul>